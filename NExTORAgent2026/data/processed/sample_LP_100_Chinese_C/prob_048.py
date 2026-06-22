import gurobipy as gp
from gurobipy import GRB


def solve_product_distribution(
    HubInventoryRequirement=300,
    StoreStandardProducts=[150, 250, 300],
    StorePremiumProducts=[30, 100, 70],
    ShippingDistance=[[0, 12, 18], [12, 0, 15], [18, 15, 0]],
    ExtraDistancePerDC=80,
    ExtraDistanceThreshold=250
):
    """
    Solves the product distribution optimization problem with:
    1) Equal hub inventory,
    2) Equal premium products per hub,
    3) Piecewise/conditional extra distance cost per DC if total shipped > threshold.
    """
    # Create a new model
    model = gp.Model("Product Distribution Optimization with Nonlinear Policy (Linearized)")

    # Parameters and Sets
    DistributionCenterNum = len(StoreStandardProducts)
    FulfillmentHubNum = len(ShippingDistance)
    ProductTypeNum = 2  # 0 for standard, 1 for premium

    DCs = range(DistributionCenterNum)
    Hubs = range(FulfillmentHubNum)
    Products = range(ProductTypeNum)

    # Pre-check: total inventory must match total hub requirements
    total_standard = sum(StoreStandardProducts)
    total_premium = sum(StorePremiumProducts)
    total_hub_demand = HubInventoryRequirement * FulfillmentHubNum

    if total_standard + total_premium != total_hub_demand:
        raise ValueError(
            f"Total inventory ({total_standard + total_premium}) "
            f"must equal total required hub inventory ({total_hub_demand})."
        )
    print(total_premium)
    print(FulfillmentHubNum)


    PremiumPerHub = total_premium // FulfillmentHubNum
    print("PremiumPerHub=", PremiumPerHub)
    # Decision Variables
    ShipAmount = model.addVars(
        DCs, Hubs, Products,
        vtype=GRB.INTEGER,
        name="ShipAmount"
    )

    # Binary variable for each DC to indicate whether extra distance cost is triggered
    DCOverThreshold = model.addVars(
        DCs,
        vtype=GRB.BINARY,
        name="DCOverThreshold"
    )

    # Total quantity shipped from each DC (all hubs, both product types)
    TotalFromDC = model.addVars(
        DCs,
        vtype=GRB.INTEGER,
        name="TotalFromDC"
    )

    # Big-M for linearization: maximum possible shipment from one DC
    # An upper bound is the inventory of that DC.
    MaxFromDC = [StoreStandardProducts[d] + StorePremiumProducts[d] for d in DCs]

    # ❤ Non-linearity is introduced. ❤
    # Original objective (commented out) – only pure distance without extra fixed cost per DC:
    # model.setObjective(
    #     gp.quicksum(ShippingDistance[d][h] * ShipAmount[d, h, p]
    #                 for d in DCs for h in Hubs for p in Products),
    #     GRB.MINIMIZE)

    # New objective: base distance + extra 80 km per DC that exceeds the threshold
    model.setObjective(
        gp.quicksum(
            ShippingDistance[d][h] * ShipAmount[d, h, p]
            for d in DCs for h in Hubs for p in Products
        )
        + gp.quicksum(
            ExtraDistancePerDC * DCOverThreshold[d] for d in DCs
        ),
        GRB.MINIMIZE
    )

    # Constraint 0: Define TotalFromDC as the sum of all shipments from each DC
    model.addConstrs(
        (
            TotalFromDC[d] ==
            gp.quicksum(ShipAmount[d, h, p] for h in Hubs for p in Products)
            for d in DCs
        ),
        name="Total_From_DC"
    )

    # Constraint 0.1: Linearization of the "if total > threshold then extra distance"
    # Threshold logic (using standard big-M formulation):
    #   TotalFromDC[d] <= Threshold + M*(DCOverThreshold[d])          (1)
    #   TotalFromDC[d] >= Threshold + 1 - M*(1 - DCOverThreshold[d])  (2)
    # This enforces:
    #   - If DCOverThreshold[d] = 0 => TotalFromDC[d] <= Threshold
    #   - If DCOverThreshold[d] = 1 => TotalFromDC[d] >= Threshold + 1
    for d in DCs:
        M = MaxFromDC[d]

        # TotalFromDC[d] <= threshold when DCOverThreshold[d] == 0
        model.addConstr(
            TotalFromDC[d] <= ExtraDistanceThreshold + M * DCOverThreshold[d],
            name=f"Threshold_Upper_DC{d + 1}"
        )

        # TotalFromDC[d] >= threshold + 1 when DCOverThreshold[d] == 1
        model.addConstr(
            TotalFromDC[d] >= (ExtraDistanceThreshold + 1)
            - M * (1 - DCOverThreshold[d]),
            name=f"Threshold_Lower_DC{d + 1}"
        )

    # Constraint 1: Fulfillment hub inventory constraint (total units per hub)
    model.addConstrs(
        (
            gp.quicksum(ShipAmount[d, h, p] for d in DCs for p in Products)
            == HubInventoryRequirement
            for h in Hubs
        ),
        name="Hub_Inventory"
    )

    # Constraint 2: Equal premium products per hub
    # Each hub must receive the same number of premium products
    model.addConstrs(
        (
            gp.quicksum(ShipAmount[d, h, 1] for d in DCs)
            >= PremiumPerHub
            for h in Hubs
        ),
        name="Equal_Premium_Per_Hub"
    )

    # Constraint 3: Inventory capacity at each DC (standard and premium separately)
    # Sum over all hubs for each DC and product type cannot exceed DC's inventory.
    model.addConstrs(
        (
            gp.quicksum(ShipAmount[d, h, 0] for h in Hubs)
            <= StoreStandardProducts[d]
            for d in DCs
        ),
        name="Standard_Inventory_Limit_DC"
    )

    model.addConstrs(
        (
            gp.quicksum(ShipAmount[d, h, 1] for h in Hubs)
            <= StorePremiumProducts[d]
            for d in DCs
        ),
        name="Premium_Inventory_Limit_DC"
    )

    # Optimize the model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        result = {
            "status": "optimal",
            "obj": model.ObjVal,
            "ShipAmount": {
                (int(d), int(h), int(p)): int(ShipAmount[d, h, p].X)
                for d in DCs for h in Hubs for p in Products
            },
            "DCOverThreshold": {
                int(d): int(DCOverThreshold[d].X) for d in DCs
            },
            "TotalFromDC": {
                int(d): int(TotalFromDC[d].X) for d in DCs
            }
        }
        return result
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_product_distribution()
    print(result)
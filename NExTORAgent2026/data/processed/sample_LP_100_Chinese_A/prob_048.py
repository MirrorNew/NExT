import gurobipy as gp
from gurobipy import GRB


def solve_product_distribution(
):
    HubInventoryRequirement = 300
    StoreStandardProducts = [150, 250, 300]
    StorePremiumProducts = [30, 100, 70]
    ShippingDistance = [[0, 12, 18], [12, 0, 15], [18, 15, 0]]
    """
    Solves the product distribution optimization problem with a
    nonlinear transportation distance cost:
    For each path (d, h), the effective distance cost is
        ( distance[d,h] * total_amount_on_path )^1.2
    where total_amount_on_path is the sum of standard and premium
    products shipped on that path.
    """
    # Create a new model
    model = gp.Model("Product Distribution Optimization (Nonlinear Distance Cost)")

    # Parameters and Sets
    DistributionCenterNum = len(StoreStandardProducts)
    FulfillmentHubNum = len(ShippingDistance)
    ProductTypeNum = 2  # 0 for standard, 1 for premium

    DCs = range(DistributionCenterNum)
    Hubs = range(FulfillmentHubNum)
    Products = range(ProductTypeNum)

    # Decision Variables: integer quantities shipped of each product type
    ShipAmount = model.addVars(
        DCs,
        Hubs,
        Products,
        vtype=GRB.INTEGER,
        name="ShipAmount"
    )

    # Auxiliary Variables: total quantity on each path (sum over product types)
    PathTotal = model.addVars(
        DCs,
        Hubs,
        vtype=GRB.CONTINUOUS,
        name="PathTotal"
    )

    # Effective distance cost variable for each path: (distance * PathTotal)^1.2
    PathCost = model.addVars(
        DCs,
        Hubs,
        vtype=GRB.CONTINUOUS,
        name="PathCost"
    )

    # Link PathTotal to ShipAmount
    model.addConstrs(
        (PathTotal[d, h] == gp.quicksum(ShipAmount[d, h, p] for p in Products)
         for d in DCs for h in Hubs),
        name="PathTotal_Definition"
    )

    # Enforce hub inventory requirement: each hub must have exactly HubInventoryRequirement items
    model.addConstrs(
        (gp.quicksum(ShipAmount[d, h, p] for d in DCs for p in Products) ==
         HubInventoryRequirement for h in Hubs),
        name="Hub_Inventory"
    )

    # NOTE: The original code only limited flow per path by DC stock.
    # If you need exact DC stock conservation, you would replace these
    # with equalities summing over hubs. Here we keep the original logic.

    # Constraint: Product type balance constraint (standard products)
    # This constraint limits the amount shipped on each individual path
    # based on the total stock at the distribution center.
    model.addConstrs(
        (ShipAmount[d, h, 0] <= StoreStandardProducts[d] for d in DCs
         for h in Hubs),
        name="Standard_Product_Limit"
    )

    # Constraint: Product type balance constraint (premium products)
    model.addConstrs(
        (ShipAmount[d, h, 1] <= StorePremiumProducts[d] for d in DCs
         for h in Hubs),
        name="Premium_Product_Limit"
    )

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(
    #     gp.quicksum(ShippingDistance[d][h] * ShipAmount[d, h, p]
    #                 for d in DCs for h in Hubs for p in Products),
    #     GRB.MINIMIZE)

    # New nonlinear objective:
    # PathCost[d,h] = (ShippingDistance[d][h] * PathTotal[d,h])^1.2
    # and we minimize sum_{d,h} PathCost[d,h]

    Y = model.addVars(
        DCs,
        Hubs,
        vtype=GRB.CONTINUOUS,
        name="PathCost"
    )

    # Define PathCost via a nonlinear power relation using Gurobi's general constraints
    for d in DCs:
        for h in Hubs:
            dist = ShippingDistance[d][h]
            # If distance is zero, cost should be zero regardless of quantity
            if dist == 0:
                model.addConstr(PathCost[d, h] == 0.0, name=f"ZeroDistCost_{d}_{h}")
            else:
                # Use a general constraint: PathCost[d,h] = (dist * PathTotal[d,h])^1.2
                # Gurobi supports power constraints: z = x^a for a in [0,2], x >= 0
                model.addConstr(Y[d,h] == dist * PathTotal[d, h])
                model.addGenConstrPow(
                    Y[d,h],
                    PathCost[d, h],
                    1.2,
                    name=f"PathCostDef_{d}_{h}"
                )

    # Nonlinear objective: minimize total effective distance cost
    model.setObjective(
        gp.quicksum(PathCost[d, h] for d in DCs for h in Hubs),
        GRB.MINIMIZE
    )

    # Optimize the model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal
        }
    else:
        return {
            "status": f"{model.status}"
        }


if __name__ == "__main__":
    result = solve_product_distribution()
    print(result)
import gurobipy as gp
from gurobipy import GRB


def solve_product_manufacturing_optimization(
    ProductPerRawMaterial=[2, 3, 2, 3],
    LaborHoursPerRawMaterial=2,
    LaborHoursPerProduct=[0, 0, 3, 4],
    SellingPrice=[10, 12, 20, 25],
    ProcessingCost=[0, 0, 5, 6],
    RawMaterialCost=5,
    MaxLaborHours=8000,
    MaxRawMaterial=3000,
    FixedCostThreshold=2000,
    ExtraFixedCost=10000
):
    """
    Models and solves the product manufacturing optimization problem,
    including a conditional fixed cost that is incurred if raw material
    purchased exceeds a given threshold (FixedCostThreshold).
    """
    # Create a new model
    model = gp.Model("Product Manufacturing Optimization with Conditional Fixed Cost")

    # Sets
    Products = range(len(SellingPrice))

    # Decision Variables
    RawMaterialPurchased = model.addVar(vtype=GRB.INTEGER, name="RawMaterialPurchased")
    ProductProduced = model.addVars(Products, vtype=GRB.INTEGER, name="ProductProduced")

    # ❤ Non-linearity is introduced. ❤
    # Original model had no conditional fixed cost term in the objective.
    # revenue = gp.quicksum(SellingPrice[p] * ProductProduced[p] for p in Products)
    # raw_material_cost = RawMaterialCost * RawMaterialPurchased
    # processing_cost = gp.quicksum(ProcessingCost[p] * ProductProduced[p] for p in Products)
    #
    # model.setObjective(revenue - raw_material_cost - processing_cost, GRB.MAXIMIZE)

    # Binary variable to activate extra fixed cost if raw material > threshold
    y = model.addVar(vtype=GRB.BINARY, name="ExtraFixedCostIndicator")

    # Objective: Maximize profit with conditional fixed cost
    revenue = gp.quicksum(SellingPrice[p] * ProductProduced[p] for p in Products)
    raw_material_cost = RawMaterialCost * RawMaterialPurchased
    processing_cost = gp.quicksum(ProcessingCost[p] * ProductProduced[p] for p in Products)
    extra_fixed_cost = ExtraFixedCost * y

    model.setObjective(
        revenue - raw_material_cost - processing_cost - extra_fixed_cost,
        GRB.MAXIMIZE
    )

    # Constraint 1: Labor hours constraint
    labor_hours = (LaborHoursPerRawMaterial * RawMaterialPurchased +
                   gp.quicksum(LaborHoursPerProduct[p] * ProductProduced[p] for p in Products))
    model.addConstr(labor_hours <= MaxLaborHours, "LaborHours")

    # Constraint 2: Raw material upper bound
    model.addConstr(RawMaterialPurchased <= MaxRawMaterial, "RawMaterial")

    # Constraint 3: Production balance constraint
    model.addConstr(
        gp.quicksum(ProductProduced[p] for p in Products) ==
        gp.quicksum(ProductPerRawMaterial[p] * RawMaterialPurchased for p in Products),
        "ProductionBalance"
    )

    # ❤ Non-linearity is introduced. ❤
    # Implement the conditional fixed cost using a binary variable and big-M logic:
    # If RawMaterialPurchased <= FixedCostThreshold -> y = 0 (no extra fixed cost)
    # If RawMaterialPurchased >  FixedCostThreshold -> y = 1 (extra fixed cost applied)
    #
    # Use a sufficiently large M. Here we can base M on MaxRawMaterial.
    M = MaxRawMaterial

    # Ensure: RawMaterialPurchased <= FixedCostThreshold + M * y
    # When y = 0 -> RawMaterialPurchased <= FixedCostThreshold
    # When y = 1 -> RawMaterialPurchased <= FixedCostThreshold + M (non-binding upper link)
    model.addConstr(
        RawMaterialPurchased <= FixedCostThreshold + M * y,
        "ConditionalFixedCost_Upper"
    )

    # Ensure: RawMaterialPurchased >= FixedCostThreshold + 1 - M * (1 - y)
    # When y = 0 -> RawMaterialPurchased >= FixedCostThreshold + 1 - M
    #              (effectively no lower restriction due to large -M term)
    # When y = 1 -> RawMaterialPurchased >= FixedCostThreshold + 1
    model.addConstr(
        RawMaterialPurchased >= FixedCostThreshold + 1 - M * (1 - y),
        "ConditionalFixedCost_Lower"
    )

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "RawMaterialPurchased": RawMaterialPurchased.X,
            "ProductProduced": {p: ProductProduced[p].X for p in Products},
            "ExtraFixedCostIndicator": y.X
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_product_manufacturing_optimization()
    print(result)
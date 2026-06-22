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
    MaxRawMaterial=3000
):
    """
    Models and solves the product manufacturing optimization problem
    with an additional non-linear policy constraint:
    The ratio of labor hours used for processing basic/advanced products
    (including further processing into premium products) to the labor hours
    used for processing raw materials must not exceed 3.
    """
    # Create a new model
    model = gp.Model("Product Manufacturing Optimization")

    # Sets
    Products = range(len(SellingPrice))

    # Decision Variables
    RawMaterialPurchased = model.addVar(vtype=GRB.INTEGER, name="RawMaterialPurchased")
    ProductProduced = model.addVars(Products, vtype=GRB.INTEGER, name="ProductProduced")

    # Objective: Maximize profit
    revenue = gp.quicksum(SellingPrice[p] * ProductProduced[p] for p in Products)
    raw_material_cost = RawMaterialCost * RawMaterialPurchased
    processing_cost = gp.quicksum(ProcessingCost[p] * ProductProduced[p] for p in Products)

    model.setObjective(revenue - raw_material_cost - processing_cost, GRB.MAXIMIZE)

    # Constraint 1: Labor hours constraint
    labor_hours = (
        LaborHoursPerRawMaterial * RawMaterialPurchased
        + gp.quicksum(LaborHoursPerProduct[p] * ProductProduced[p] for p in Products)
    )
    model.addConstr(labor_hours <= MaxLaborHours, "LaborHours")

    # Constraint 2: Raw material constraint
    model.addConstr(RawMaterialPurchased <= MaxRawMaterial, "RawMaterial")

    # Constraint 3: Product production constraint
    model.addConstr(
        gp.quicksum(ProductProduced[p] for p in Products)
        == gp.quicksum(ProductPerRawMaterial[p] * RawMaterialPurchased for p in Products),
        "ProductionBalance",
    )

    # ❤ Non-linearity is introduced. ❤
    # New non-linear policy constraint:
    # Ratio of labor hours for processing basic/advanced (including premium)
    # to labor hours for processing raw material must not exceed 3:
    #    (Σ_p LaborHoursPerProduct[p] * ProductProduced[p]) /
    #    (LaborHoursPerRawMaterial * RawMaterialPurchased) <= 3
    #
    # This is equivalent to:
    #    Σ_p LaborHoursPerProduct[p] * ProductProduced[p]
    #        <= 3 * LaborHoursPerRawMaterial * RawMaterialPurchased
    #
    # The term on the right-hand side involves the product of a variable and
    # a linear expression, which we model explicitly as a non-linear constraint.
    labor_for_products = gp.quicksum(
        LaborHoursPerProduct[p] * ProductProduced[p] for p in Products
    )
    labor_for_raw_material = LaborHoursPerRawMaterial * RawMaterialPurchased

    # Add the non-linear ratio constraint using Gurobi's general constraint interface
    # to emphasize the introduced non-linearity between decision variables.
    model.addQConstr(
        labor_for_products,
        GRB.LESS_EQUAL,
        3 * labor_for_raw_material,
        name="PolicyRatioConstraint",
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
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_product_manufacturing_optimization()
    print(result)
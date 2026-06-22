import gurobipy as gp
from gurobipy import GRB


def solve_product_manufacturing_optimization(
):
    ProductPerRawMaterial=[2, 3, 0, 0]  # Basic, Advanced, Premium Basic, Premium Advanced directly from raw
    LaborHoursPerRawMaterial=2
    LaborHoursPerProduct=[0, 0, 3, 4] # Extra labor for upgrading Basic->PremBasic, Adv->PremAdv
    SellingPrice=[10, 12, 20, 25]    # Basic, Advanced, PremBasic, PremAdv
    ProcessingCost=[0, 0, 5, 6]      # Upgrade cost Basic->PremBasic, Adv->PremAdv
    RawMaterialCost=5
    MaxLaborHours=8000
    MaxRawMaterial=3000
    """
    Models and solves the (now non-linear) product manufacturing optimization problem
    with a multiplicative synergy term.
    """
    # Create a new model
    model = gp.Model("Product Manufacturing Optimization with Synergy")

    # Sets
    Products = range(len(SellingPrice))

    # Indices:
    # 0: Basic
    # 1: Advanced
    # 2: Premium Basic (from Basic)
    # 3: Premium Advanced (from Advanced)

    # Decision Variables
    RawMaterialPurchased = model.addVar(vtype=GRB.INTEGER, name="RawMaterialPurchased")
    ProductProduced = model.addVars(Products, vtype=GRB.INTEGER, name="ProductProduced")

    # For clarity, define aliases
    Basic = 0
    Advanced = 1
    PremBasic = 2
    PremAdvanced = 3

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # revenue = gp.quicksum(SellingPrice[p] * ProductProduced[p] for p in Products)
    # raw_material_cost = RawMaterialCost * RawMaterialPurchased
    # processing_cost = gp.quicksum(ProcessingCost[p] * ProductProduced[p] for p in Products)
    # model.setObjective(revenue - raw_material_cost - processing_cost, GRB.MAXIMIZE)

    # New components for objective with synergy:
    # Revenue from selling all (non-upgraded + upgraded) products
    revenue = gp.quicksum(SellingPrice[p] * ProductProduced[p] for p in Products)

    # Raw material cost
    raw_material_cost = RawMaterialCost * RawMaterialPurchased

    # Further processing (upgrade) cost: only applies to Premium products
    processing_cost = (
        ProcessingCost[PremBasic] * ProductProduced[PremBasic]
        + ProcessingCost[PremAdvanced] * ProductProduced[PremAdvanced]
    )

    # Synergy term (non-linear):
    # 协同利润 = 0.01 ×（用于生产基础型产品的原材料数量）×
    #                    （进一步加工为高端基础型产品的基础型产品数量）×
    #                    （进一步加工为高端高级型产品的高级型产品数量）
    #
    # Interpretation consistent with variable definitions:
    # - RawMaterialPurchased: total raw material purchased
    # - ProductProduced[PremBasic]: number of Premium Basic actually produced
    # - ProductProduced[PremAdvanced]: number of Premium Advanced actually produced
    #
    # We model synergy as:
    # synergy_profit = 0.01 * RawMaterialPurchased * ProductProduced[PremBasic] * ProductProduced[PremAdvanced]

    Y = model.addVar()
    model.addConstr(Y == ProductProduced[PremBasic] * ProductProduced[PremAdvanced])
    synergy_profit = 0.0001 * RawMaterialPurchased * Y
    # Total profit = revenue - material cost - processing cost + synergy profit
    model.setObjective(revenue - raw_material_cost - processing_cost + synergy_profit, GRB.MAXIMIZE)

    # Constraint 1: Labor hours constraint
    labor_hours = (
        LaborHoursPerRawMaterial * RawMaterialPurchased +
        gp.quicksum(LaborHoursPerProduct[p] * ProductProduced[p] for p in Products)
    )
    model.addConstr(labor_hours <= MaxLaborHours, "LaborHours")

    # Constraint 2: Raw material quantity constraint
    model.addConstr(RawMaterialPurchased <= MaxRawMaterial, "RawMaterial")

    # Constraint 3: Production balance for Basic and Advanced from raw material
    # Each unit of raw material can produce:
    #   2 units Basic and 3 units Advanced
    # A simple way is to ensure that output of Basic and Advanced does not exceed
    # what can be produced from raw material.
    model.addConstr(
        ProductProduced[Basic] <= ProductPerRawMaterial[Basic] * RawMaterialPurchased,
        "BasicFromRaw"
    )
    model.addConstr(
        ProductProduced[Advanced] <= ProductPerRawMaterial[Advanced] * RawMaterialPurchased,
        "AdvancedFromRaw"
    )

    # Constraint 4: Upgrading consistency
    # Premium Basic can only be produced by upgrading Basic
    model.addConstr(
        ProductProduced[PremBasic] <= ProductProduced[Basic],
        "PremBasicFromBasic"
    )
    # Premium Advanced can only be produced by upgrading Advanced
    model.addConstr(
        ProductProduced[PremAdvanced] <= ProductProduced[Advanced],
        "PremAdvancedFromAdvanced"
    )

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "RawMaterialPurchased": RawMaterialPurchased.X,
            "ProductProduced": {p: ProductProduced[p].X for p in Products}
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_product_manufacturing_optimization()
    print(result)
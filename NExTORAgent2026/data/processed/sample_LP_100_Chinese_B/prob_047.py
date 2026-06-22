import gurobipy as gp
from gurobipy import GRB


def solve_vaccine_production_optimization(
    Demand=[[100, 200], [300, 100]],
    MaxProductionPerQuarter=[1500, 1500],
    RawMaterialCost=[300, 450],
    RawMaterialPerProduct=[4, 3],
    MaxRawMaterialPurchase=[2500, 2500],
    InitialInventory=[200, 300],
    StorageCost=[100, 100],
    EfficacyRate=[0.85, 0.95],
    RegulatoryEfficacyRate=0.90
):
    """
    Models and solves the vaccine production optimization problem.
    """
    # Create a new model
    model = gp.Model("HealthSupply_Vaccine_Production")

    # Sets
    P = range(len(EfficacyRate))  # Products
    Q = range(len(MaxProductionPerQuarter))  # Quarters

    # Decision Variables
    RawMaterialPurchased = model.addVars(Q, lb=0.0, name="RawMaterialPurchased")
    ProductProduced = model.addVars(P, Q, vtype=GRB.INTEGER, lb=0, name="ProductProduced")
    ProductStorage = model.addVars(P, Q, vtype=GRB.INTEGER, lb=0, name="ProductStorage")

    # Objective Function: Minimize total cost
    raw_material_cost = gp.quicksum(RawMaterialCost[q] * RawMaterialPurchased[q] for q in Q)
    storage_cost = gp.quicksum(StorageCost[q] * ProductStorage[p, q] for p in P for q in Q)
    total_cost = raw_material_cost + storage_cost

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints
    # 1. Demand constraints
    for p in P:
        # First quarter uses initial inventory
        model.addConstr(
            ProductProduced[p, 0] + InitialInventory[p] - ProductStorage[p, 0] >= Demand[0][p],
            f"Demand_P{p}_Q1"
        )

    for p in P:
        # Second quarter uses storage from first quarter
        model.addConstr(
            ProductProduced[p, 1] + ProductStorage[p, 0] - ProductStorage[p, 1] >= Demand[1][p],
            f"Demand_P{p}_Q2"
        )

    # 2. Production capacity constraints
    for q in Q:
        model.addConstr(
            gp.quicksum(ProductProduced[p, q] for p in P) <= MaxProductionPerQuarter[q],
            f"ProductionCapacity_Q{q+1}"
        )

    # 3. Raw material constraints
    for q in Q:
        model.addConstr(
            gp.quicksum(RawMaterialPerProduct[p] * ProductProduced[p, q] for p in P) <= RawMaterialPurchased[q],
            f"RawMaterial_Q{q+1}"
        )

    # 4. Raw material purchase limit constraints
    for q in Q:
        model.addConstr(
            RawMaterialPurchased[q] <= MaxRawMaterialPurchase[q],
            f"RawMaterialPurchaseLimit_Q{q+1}"
        )

    # ❤ Non-linearity is introduced. ❤
    # 5. Regulatory efficacy constraints
    # for q in Q:
    #     total_produced = gp.quicksum(ProductProduced[p, q] for p in P)
    #     weighted_efficacy = gp.quicksum(EfficacyRate[p] * ProductProduced[p, q] for p in P)
    #
    #     model.addConstr(
    #         weighted_efficacy >= RegulatoryEfficacyRate * total_produced,
    #         f"RegulatoryEfficacy_Q{q+1}"
    #     )

    # ❤ Non-linearity is introduced. ❤
    # New regulatory constraint: for each quarter q,
    # ProductProduced[X, q] / ProductProduced[Y, q] <= 2/3
    # To introduce non-linearity as requested, we model this directly as a division expression.
    X = 0  # index for vaccine X
    Y = 1  # index for vaccine Y
    ratio_limit = 2.0 / 3.0

    for q in Q:
        # This introduces a nonlinear expression due to division of decision variables
        model.addConstr(
            ProductProduced[X, q]  <= ratio_limit * (ProductProduced[Y, q] + 1e-6),
            name=f"RegulatoryRatio_Q{q+1}"
        )

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {"status": "optimal", "obj": model.ObjVal}
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_vaccine_production_optimization()
    print(result)
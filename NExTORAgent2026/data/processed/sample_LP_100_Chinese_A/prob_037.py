import math

import gurobipy as gp
from gurobipy import GRB


def solve_apple_processing_optimization(
):
    Inventory=[90000, 110000]
    Revenue=[0.35, 1.10]
    VariableCost=[0.12, 0.80]
    AverageGrade=[5, 6]
    MaxProcessingCapacity=150000
    MinPackagePercentage=0.20
    Grades=[7, 4]
    # Create a new model
    model = gp.Model("Apple Processing Optimization")

    # Sets
    Products = range(len(Revenue))

    # Profit per unit for each product
    Profit = [Revenue[p] - VariableCost[p] for p in Products]

    # Create decision variables dictionary
    ProcessedAmount = {}
    for p in Products:
        for g_idx, g in enumerate(Grades):
            ProcessedAmount[p, g_idx] = model.addVar(
                vtype=GRB.CONTINUOUS,
                name=f"ProcessedAmount_{p+1}_{g}"
            )

    # Objective: Maximize profit
    obj = gp.quicksum(
        Profit[p] * ProcessedAmount[p, g_idx]
        for p in Products
        for g_idx in range(len(Grades))
    )
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint 1: Inventory constraint
    for g_idx in range(len(Grades)):
        model.addConstr(
            gp.quicksum(ProcessedAmount[p, g_idx] for p in Products) <= Inventory[g_idx],
            f"Inventory_{Grades[g_idx]}"
        )

    # Constraint 2: Average grade constraint
    for p in Products:
        model.addConstr(
            AverageGrade[p] * gp.quicksum(ProcessedAmount[p, g_idx] for g_idx in range(len(Grades))) <=
            gp.quicksum(Grades[g_idx] * ProcessedAmount[p, g_idx] for g_idx in range(len(Grades))),
            f"AverageGrade_{p+1}"
        )

    # Constraint 3: Processing capacity constraint
    total_processed = gp.quicksum(
        ProcessedAmount[p, g_idx] for p in Products for g_idx in range(len(Grades))
    )
    model.addConstr(
        total_processed <= MaxProcessingCapacity,
        "ProcessingCapacity"
    )

    # ❤ Non-linearity is introduced. ❤
    # Constraint 4: Nonlinear marketing constraint with exponential preference
    # x = total amount of apples used for packaged fresh fruit (product 0)
    x_packaged = gp.quicksum(ProcessedAmount[0, g_idx] for g_idx in range(len(Grades)))

    # The required share of packaged apples is:
    # share_required = 0.2 * 1.3^(x_packaged / 100000)
    # Constraint: x_packaged >= share_required * total_processed
    # share_required = MinPackagePercentage * (1.3 ** (x_packaged / 100000.0))

    Y = model.addVar()
    model.addConstr(Y == x_packaged / 10000.0)
    # share_required = MinPackagePercentage * (1.3 ** Y)
    # ln share_required = ln MinPackagePercentage + Y * ln 1.3
    ln_share_required = model.addVar()
    model.addConstr(ln_share_required == math.log(MinPackagePercentage)  + Y * math.log(1.3))
    Y_share_required = model.addVar()
    model.addGenConstrLog(Y_share_required, ln_share_required)

    model.addConstr(
        x_packaged >= Y_share_required * total_processed,
        "Marketing_Nonlinear"
    )

    # Optimize the model
    # Note: A nonlinear constraint is present, so we must use a non-linear capable license of Gurobi.
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        return {"status": "optimal", "obj": model.objVal}
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_apple_processing_optimization()
    print(result)
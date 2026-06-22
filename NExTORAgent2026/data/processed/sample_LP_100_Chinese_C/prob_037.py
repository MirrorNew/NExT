import gurobipy as gp
from gurobipy import GRB


def solve_apple_processing_optimization(
    Inventory=[90000, 110000],
    Revenue=[0.35, 1.10],
    VariableCost=[0.12, 0.80],
    AverageGrade=[5, 6],
    MaxProcessingCapacity=150000,
    MinPackagePercentage=0.20,
    Grades=[7, 4]
):
    # Create a new model
    model = gp.Model("Apple Processing Optimization")

    # Sets
    Products = range(len(Revenue))  # 0: package, 1: juice

    # Profit per unit for each product (base variable cost part only)
    Profit = [Revenue[p] - VariableCost[p] for p in Products]

    # Create decision variables dictionary
    ProcessedAmount = {}
    for p in Products:
        for g_idx, g in enumerate(Grades):
            ProcessedAmount[p, g_idx] = model.addVar(
                vtype=GRB.CONTINUOUS,
                name=f"ProcessedAmount_{p+1}_{g}"
            )

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # obj = gp.quicksum(
    #     Profit[p] * ProcessedAmount[p, g_idx]
    #     for p in Products
    #     for g_idx in range(len(Grades))
    # )

    # --- New non-linear (actually MILP) objective with step cost for juice ---
    # Auxiliary variables for total amounts
    total_package = gp.quicksum(ProcessedAmount[0, g_idx] for g_idx in range(len(Grades)))
    total_juice = gp.quicksum(ProcessedAmount[1, g_idx] for g_idx in range(len(Grades)))

    # Binary variable indicating whether juice production exceeds 80,000 lbs
    y_overtime = model.addVar(vtype=GRB.BINARY, name="JuiceOvertimeIndicator")

    # Fixed overtime cost if juice production > 80,000 lbs
    FixedOvertimeCost = 10000.0
    JuiceThreshold = 80000.0

    # Big-M for linking binary and total_juice; max feasible juice is MaxProcessingCapacity
    M = MaxProcessingCapacity

    # If y_overtime = 0  =>  total_juice <= 80,000
    model.addConstr(total_juice <= JuiceThreshold + M * y_overtime,
                    name="JuiceThresholdUpper")
    # If total_juice > 80,000 (to justify paying fixed cost) then y_overtime must be 1.
    # A small epsilon can be used; here we ensure consistency by:
    model.addConstr(total_juice >= JuiceThreshold * y_overtime,
                    name="JuiceThresholdLower")

    # Profit expression (without fixed cost)
    linear_profit = (
        Profit[0] * total_package +
        Profit[1] * total_juice
    )

    # Subtract fixed overtime cost when y_overtime = 1
    obj = linear_profit - FixedOvertimeCost * y_overtime

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
    model.addConstr(
        gp.quicksum(ProcessedAmount[p, g_idx] for p in Products for g_idx in range(len(Grades))) <= MaxProcessingCapacity,
        "ProcessingCapacity"
    )

    # Constraint 4: Marketing constraint (at least 20% must be packaged)
    model.addConstr(
        total_package >=
        MinPackagePercentage * gp.quicksum(ProcessedAmount[p, g_idx]
                                           for p in Products
                                           for g_idx in range(len(Grades))),
        "Marketing"
    )

    # Optimize the model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.objVal
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_apple_processing_optimization()
    print(result)
import math

import gurobipy as gp
from gurobipy import GRB


def solve_staffing_optimization():
    FixedCost = [1000, 2000]
    StaffCost = [500, 900]
    MaxStaffNum = [7, 7]
    ProcessRate = [[20, 25], [18, 22], [15, 20]]
    Demand = [120, 150, 200]
    """
    Solves the distribution center staffing optimization problem.
    """
    # --- 1. Model Creation ---
    model = gp.Model("Distribution Center Staffing (Nonlinear)")

    # --- 2. Parameters & Sets ---
    Centers = range(len(FixedCost))
    GoodsTypes = range(len(Demand))

    # --- 3. Decision Variables ---
    # ActivatingCenter[c] = 1 if center c is activated, 0 otherwise
    ActivatingCenter = model.addVars(Centers,
                                     vtype=GRB.BINARY,
                                     name="ActivatingCenter")

    # StaffNum[c] = number of staff at center c
    StaffNum = model.addVars(Centers, vtype=GRB.INTEGER, name="StaffNum")

    # --- 4. Objective Function ---
    # Minimize total cost (fixed costs + staff costs)
    obj = gp.quicksum(FixedCost[c] * ActivatingCenter[c]
                      + StaffCost[c] * StaffNum[c] for c in Centers)
    model.setObjective(obj, GRB.MINIMIZE)

    # --- 5. Constraints ---
    # Constraint 1: Staff number constraint
    model.addConstrs(
        (StaffNum[c] <= MaxStaffNum[c] * ActivatingCenter[c] for c in Centers),
        name="StaffLimit")

    # ❤ Non-linearity is introduced. ❤
    # Original linear demand-satisfaction constraints are commented out:
    # model.addConstrs(
    #     (gp.quicksum(ProcessRate[g][c] * StaffNum[c] for c in Centers) >=
    #      Demand[g] for g in GoodsTypes),
    #     name="DemandSatisfaction")

    # --- 5a. Nonlinear demand satisfaction constraints ---

    # For type-1 goods, center 1 has exponential productivity:
    # If x staff at center 1 => capacity = 20 * x * 1.3^x
    # Center 2 remains linear with rate ProcessRate[0][1] = 25
    x1 = StaffNum[0]
    x2 = StaffNum[1]


    pow_13_x1 = model.addVar()
    # pow_13_x1 == 1.3 ** x1
    ln_pow_13_x1 = model.addVar()
    model.addGenConstrLog(pow_13_x1,ln_pow_13_x1)
    model.addConstr(ln_pow_13_x1 == x1 * math.log(1.3))

    # Type-1 demand with nonlinear term for center 1
    model.addConstr(
        20 * x1 * pow_13_x1 + ProcessRate[0][1] * x2 >= Demand[0],
        name="Demand_Type1_Nonlinear"
    )

    # Type-2 and Type-3 demands remain linear as in the original model
    # Type-2: 18 * x1 + 22 * x2 >= 150
    model.addConstr(
        ProcessRate[1][0] * x1 + ProcessRate[1][1] * x2 >= Demand[1],
        name="Demand_Type2"
    )

    # Type-3: 15 * x1 + 20 * x2 >= 200
    model.addConstr(
        ProcessRate[2][0] * x1 + ProcessRate[2][1] * x2 >= Demand[2],
        name="Demand_Type3"
    )

    # --- 6. Solve the Model ---
    # This is now a Mixed-Integer Nonlinear Program (MINLP)
    model.Params.NonConvex = 2  # allow non-convex nonlinearities such as a^x
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "StaffNum": {c: int(StaffNum[c].X) for c in Centers},
            "ActivatingCenter": {c: int(ActivatingCenter[c].X) for c in Centers},
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_staffing_optimization()
    print(result)
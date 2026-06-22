import gurobipy as gp
from gurobipy import GRB


def solve_staffing_optimization(
    FixedCost=[1000, 2000],
    StaffCost=[500, 900],
    MaxStaffNum=[7, 7],
    ProcessRate=[[20, 25], [18, 22], [15, 20]],
    Demand=[120, 150, 0]):
    """
    Solves the distribution center staffing optimization problem.

    Non-linear change (Type B) in the problem description:
    - Original requirement: At least 200 units of type 3 goods.
    - New requirement: At least 120 units of type 1 goods, at least 150 units of type 2 goods,
      and the ratio of type 3 to type 2 goods processed must be at least 4/3.

    The original type-3 minimum demand of 200 is removed and replaced with:
        total_type3 >= (4/3) * total_type2
    """

    # --- 1. Model Creation ---
    model = gp.Model("Distribution Center Staffing")

    # --- 2. Parameters & Sets ---
    Centers = range(len(FixedCost))
    GoodsTypes = range(3)  # We still have 3 types of goods

    # --- 3. Decision Variables ---
    # ActivatingCenter[c] = 1 if center c is activated, 0 otherwise
    ActivatingCenter = model.addVars(Centers,
                                     vtype=GRB.BINARY,
                                     name="ActivatingCenter")

    # StaffNum[c] = number of staff at center c
    StaffNum = model.addVars(Centers, vtype=GRB.INTEGER, name="StaffNum")

    # --- 4. Objective Function ---
    # Minimize total cost (fixed costs + staff costs)
    obj = gp.quicksum(FixedCost[c] * ActivatingCenter[c] +
                      StaffCost[c] * StaffNum[c] for c in Centers)
    model.setObjective(obj, GRB.MINIMIZE)

    # --- 5. Constraints ---
    # Constraint 1: Staff number constraint
    model.addConstrs(
        (StaffNum[c] <= MaxStaffNum[c] * ActivatingCenter[c] for c in Centers),
        name="StaffLimit")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstrs(
    #     (gp.quicksum(ProcessRate[g][c] * StaffNum[c] for c in Centers) >=
    #      Demand[g] for g in GoodsTypes),
    #     name="DemandSatisfaction")

    # New constraints:
    # 1) Keep minimum demand constraints only for type 1 and type 2
    #    type 1 index g = 0, type 2 index g = 1
    model.addConstr(
        gp.quicksum(ProcessRate[0][c] * StaffNum[c] for c in Centers) >= 120,
        name="Demand_Type1_Min"
    )
    model.addConstr(
        gp.quicksum(ProcessRate[1][c] * StaffNum[c] for c in Centers) >= 150,
        name="Demand_Type2_Min"
    )

    # 2) Ratio constraint: total type 3 >= (4/3) * total type 2
    total_type2 = gp.quicksum(ProcessRate[1][c] * StaffNum[c] for c in Centers)
    total_type3 = gp.quicksum(ProcessRate[2][c] * StaffNum[c] for c in Centers)

    # This is a linear inequality because both sides are linear expressions:
    #     total_type3 >= (4/3) * total_type2
    model.addConstr(
        total_type3 * 3 >= 2 * total_type2,
        name="Type3_to_Type2_Ratio"
    )

    # --- 6. Solve the Model ---
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "ActivatingCenter": {c: int(ActivatingCenter[c].X) for c in Centers},
            "StaffNum": {c: int(StaffNum[c].X) for c in Centers},
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_staffing_optimization()
    print(result)
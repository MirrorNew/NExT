import gurobipy as gp
from gurobipy import GRB


def solve_staffing_optimization(
    FixedCost=[1000, 2000],
    StaffCost=[500, 900],
    MaxStaffNum=[7, 7],
    ProcessRate=[[20, 25], [18, 22], [15, 20]],
    Demand=[120, 150, 200],
    ExtraCostCenter2=1500,
    ExtraThresholdCenter2=5
):
    """
    Solves the distribution center staffing optimization problem with
    an additional cost rule:
    If in a given week the number of coordinators at distribution
    center 2 exceeds 5 (i.e., is 6 or 7), an extra operating cost of
    1,500 USD is incurred for that week at distribution center 2.

    The code structure keeps the original model and explicitly
    comments where non-linearity is conceptually introduced, even
    though the implementation remains MILP (using binary variables).
    """
    # --- 1. Model Creation ---
    model = gp.Model("Distribution Center Staffing with Extra Cost Rule")

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

    # ❤ Non-linearity is introduced. ❤
    # Extra indicator variable for the non-linear rule at center 2:
    # ExtraCostIndicator2 = 1 if StaffNum[1] > 5 (i.e., 6 or 7), else 0
    ExtraCostIndicator2 = model.addVar(vtype=GRB.BINARY,
                                       name="ExtraCostIndicator2")

    # --- 4. Objective Function ---
    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out and replaced by one that
    # includes the extra cost term that depends on StaffNum[1] > 5):
    # obj = gp.quicksum(FixedCost[c] * ActivatingCenter[c] +
    #                   StaffCost[c] * StaffNum[c] for c in Centers)

    # New objective including the additional 1,500 USD cost at center 2
    # when ExtraCostIndicator2 = 1
    obj = (
        gp.quicksum(
            FixedCost[c] * ActivatingCenter[c] +
            StaffCost[c] * StaffNum[c] for c in Centers
        )
        + ExtraCostCenter2 * ExtraCostIndicator2
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # --- 5. Constraints ---
    # Constraint 1: Staff number constraint
    model.addConstrs(
        (StaffNum[c] <= MaxStaffNum[c] * ActivatingCenter[c] for c in Centers),
        name="StaffLimit"
    )

    # Constraint 2: Demand satisfaction constraint
    model.addConstrs(
        (gp.quicksum(ProcessRate[g][c] * StaffNum[c] for c in Centers) >=
         Demand[g] for g in GoodsTypes),
        name="DemandSatisfaction"
    )

    # ❤ Non-linearity is introduced. ❤
    # Additional logical constraint for the non-linear extra cost rule:
    # ExtraCostIndicator2 = 1  <=>  StaffNum[1] > 5
    # We linearize this logical condition with big-M style constraints.

    c = 1  # index for center 2
    M = MaxStaffNum[c]  # big-M can be the maximum staff number (7 here)

    # If ExtraCostIndicator2 = 0, then StaffNum[1] <= 5
    model.addConstr(
        StaffNum[c] <= ExtraThresholdCenter2 + M * ExtraCostIndicator2,
        name="UpperLink_ExtraCostCenter2"
    )

    # If StaffNum[1] >= 6, then ExtraCostIndicator2 must be 1.
    # Implemented via: StaffNum[1] >= (ExtraThresholdCenter2 + 1) * ExtraCostIndicator2
    model.addConstr(
        StaffNum[c] >= (ExtraThresholdCenter2 + 1) * ExtraCostIndicator2,
        name="LowerLink_ExtraCostCenter2"
    )

    # Optionally link ExtraCostIndicator2 to ActivatingCenter[1]
    # so that no extra cost is charged if the center is not active:
    model.addConstr(
        ExtraCostIndicator2 <= ActivatingCenter[c],
        name="ActivationLink_ExtraCostCenter2"
    )

    # --- 6. Solve the Model ---
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        # Extract some key decision variables for clarity
        staff_nums = {c: StaffNum[c].X for c in Centers}
        activated = {c: ActivatingCenter[c].X for c in Centers}
        extra_cost_flag = ExtraCostIndicator2.X
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "staff_nums": staff_nums,
            "activated": activated,
            "extra_cost_center2": extra_cost_flag,
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_staffing_optimization()
    print(result)
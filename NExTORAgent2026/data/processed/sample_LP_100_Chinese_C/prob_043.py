import gurobipy as gp
from gurobipy import GRB

def solve_waste_treatment_optimization(
    TreatmentCost=[12, 8, 18],
    PollutantReduction=[
        [0.15, 0.35],
        [0.25, 0.20],
        [0.30, 0.40]
    ],
    PollutionTarget=[25, 35],
    FixedCost_Facility2=5000,
    Facility2_Index=1,   # Python index for facility 2 (since list is 0-based)
    Threshold_Facility2=40
):
    # Create a new model
    model = gp.Model("Waste Treatment Optimization with Fixed Charge")

    # Sets
    Facilities = range(len(TreatmentCost))
    Pollutants = range(len(PollutionTarget))

    # Decision Variables: treatment amount at each facility
    TreatmentAmount = model.addVars(Facilities, vtype=GRB.CONTINUOUS, name="TreatmentAmount")

    # ❤ Non-linearity is introduced. ❤
    # Original purely linear objective (commented out):
    # obj = gp.quicksum(TreatmentCost[f] * TreatmentAmount[f] for f in Facilities)
    # model.setObjective(obj, GRB.MINIMIZE)

    # Introduce a binary variable for facility 2 fixed cost (fixed-charge / step cost)
    use_fixed_cost_f2 = model.addVar(vtype=GRB.BINARY, name="UseFixedCost_F2")

    # Objective: linear variable costs + conditional fixed cost for facility 2
    obj = (
        gp.quicksum(TreatmentCost[f] * TreatmentAmount[f] for f in Facilities)
        + FixedCost_Facility2 * use_fixed_cost_f2
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Pollution reduction constraints
    for p in Pollutants:
        model.addConstr(
            gp.quicksum(PollutantReduction[f][p] * TreatmentAmount[f] for f in Facilities) >= PollutionTarget[p],
            f"PollutionReduction_{p+1}"
        )

    # ❤ Non-linearity is introduced. ❤
    # Link the binary variable with the treatment amount at facility 2.
    # If TreatmentAmount[Facility2_Index] > Threshold_Facility2, then use_fixed_cost_f2 must be 1;
    # if TreatmentAmount[Facility2_Index] <= Threshold_Facility2, model can choose use_fixed_cost_f2 = 0
    # and avoid the fixed cost, but may set it to 1 if beneficial (it won't be, given it's a cost).
    #
    # This is modeled using a "big-M" style linearization:
    #   TreatmentAmount[2] - Threshold_Facility2 <= M * use_fixed_cost_f2
    #
    # Choose a sufficiently large M to cover the feasible maximum treatment amount.
    # For illustration, we pick M = 1e3.
    M = 1e3
    model.addConstr(
        TreatmentAmount[Facility2_Index] - Threshold_Facility2 <= M * use_fixed_cost_f2,
        "Facility2_FixedCost_Trigger"
    )

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        solution = {
            "status": "optimal",
            "obj": model.ObjVal,
            "TreatmentAmount": {int(f): TreatmentAmount[f].X for f in Facilities},
            "UseFixedCost_F2": use_fixed_cost_f2.X
        }
        return solution
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_waste_treatment_optimization()
    print(result)
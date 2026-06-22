import gurobipy as gp
from gurobipy import GRB

def solve_waste_treatment_optimization(
    TreatmentCost=[12, 8, 18],
    PollutantReduction=[
        [0.15, 0.35],  # [X, Y] reductions per ton treated at facility 1
        [0.25, 0.20],  # [X, Y] reductions per ton treated at facility 2
        [0.30, 0.40]   # [X, Y] reductions per ton treated at facility 3
    ],
    PollutionTarget=[25, 35]  # [X_target, Y_target]
):
    # Create a new model
    model = gp.Model("Waste Treatment Optimization")

    # Sets
    Facilities = range(len(TreatmentCost))
    Pollutants = range(len(PollutionTarget))  # 0: X, 1: Y

    # Decision Variables
    TreatmentAmount = model.addVars(Facilities, vtype=GRB.CONTINUOUS, name="TreatmentAmount")

    # Objective: Minimize total treatment cost
    obj = gp.quicksum(TreatmentCost[f] * TreatmentAmount[f] for f in Facilities)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: Pollution reduction constraints for each pollutant
    # Original linear constraints:
    # for p in Pollutants:
    #     model.addConstr(
    #         gp.quicksum(PollutantReduction[f][p] * TreatmentAmount[f] for f in Facilities) >= PollutionTarget[p],
    #         f"PollutionReduction_{p+1}"
    #     )

    # Keep the original minimum reduction constraints for X and Y:
    # X reduction >= 25
    model.addConstr(
        gp.quicksum(PollutantReduction[f][0] * TreatmentAmount[f] for f in Facilities) >= PollutionTarget[0],
        "PollutionReduction_X_min"
    )

    # Y reduction >= 35
    model.addConstr(
        gp.quicksum(PollutantReduction[f][1] * TreatmentAmount[f] for f in Facilities) >= PollutionTarget[1],
        "PollutionReduction_Y_min"
    )

    # ❤ Non-linearity is introduced. ❤
    # New non-linear constraint: total reduction of X / total reduction of Y >= 0.7
    # This can be written as a quadratic (bilinear) inequality:
    #   sum_f rX_f * t_f  >= 0.7 * sum_f rY_f * t_f
    total_reduction_X = gp.quicksum(PollutantReduction[f][0] * TreatmentAmount[f] for f in Facilities)
    total_reduction_Y = gp.quicksum(PollutantReduction[f][1] * TreatmentAmount[f] for f in Facilities)
    model.addQConstr(total_reduction_X >= 0.7 * total_reduction_Y, "Ratio_X_to_Y_at_least_0_7")

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "TreatmentAmounts": {f: TreatmentAmount[f].X for f in Facilities},
            "TotalReductionX": total_reduction_X.getValue(),
            "TotalReductionY": total_reduction_Y.getValue()
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_waste_treatment_optimization()
    print(result)
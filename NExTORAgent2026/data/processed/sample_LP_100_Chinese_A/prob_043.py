import gurobipy as gp
from gurobipy import GRB

def solve_waste_treatment_optimization(
):
    TreatmentCost=[12, 8, 18]
    PollutantReduction=[
        [0.15, 0.35],
        [0.25, 0.20],
        [0.30, 0.40]
    ]
    PollutionTarget=[25, 35]
    # Create a new model
    model = gp.Model("Waste Treatment Optimization with Nonlinear Synergy Cost")

    # Sets
    Facilities = range(len(TreatmentCost))
    Pollutants = range(len(PollutionTarget))

    # Decision Variables
    TreatmentAmount = model.addVars(Facilities, vtype=GRB.CONTINUOUS, name="TreatmentAmount")

    # ❤ Non-linearity is introduced. ❤
    # Objective: Minimize total treatment cost (linear costs only)
    # obj = gp.quicksum(TreatmentCost[f] * TreatmentAmount[f] for f in Facilities)
    # model.setObjective(obj, GRB.MINIMIZE)

    # New objective: linear treatment cost + nonlinear synergy cost 0.002 * x1 * x2 * x3
    x1 = TreatmentAmount[0]
    x2 = TreatmentAmount[1]
    x3 = TreatmentAmount[2]

    linear_cost = gp.quicksum(TreatmentCost[f] * TreatmentAmount[f] for f in Facilities)

    Y = model.addVar(lb=0)
    model.addConstr(Y == x2 * x3)
    model.addConstr(x3 >= 1)
    synergy_cost = 0.01 * x1 * Y
    model.setObjective(linear_cost + synergy_cost, GRB.MINIMIZE)

    # Constraint 1: Pollution reduction constraints
    for p in Pollutants:
        model.addConstr(
            gp.quicksum(PollutantReduction[f][p] * TreatmentAmount[f] for f in Facilities) >= PollutionTarget[p],
            f"PollutionReduction_{p+1}"
        )

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        solution = {f"x{f+1}": TreatmentAmount[f].X for f in Facilities}
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "solution": solution
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_waste_treatment_optimization()
    print(result)
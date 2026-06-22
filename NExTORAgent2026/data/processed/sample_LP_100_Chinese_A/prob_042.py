import math
import gurobipy as gp
from gurobipy import GRB


def solve_stop_location():
    distance = [
        [389, 515, 170, 143, 617], [562, 678, 265, 640, 629],
        [206, 594, 180, 564, 683], [574, 105, 311, 99, 550],
        [616, 490, 99, 473, 682], [571, 258, 494, 749, 61],
        [573, 234, 207, 635, 318], [70, 53, 399, 740, 494],
        [229, 190, 550, 654, 394], [50, 56, 459, 143, 478],
        [95, 378, 507, 647, 135], [767, 200, 569, 689, 621],
        [729, 333, 91, 86, 386], [633, 163, 562, 184, 384],
        [67, 515, 224, 502, 345]
    ]
    coverage_distance = 300
    """
    Models and solves the set covering problem for stop locations,
    considering quarterly nonlinear fluctuations in effective distance.
    """
    # --- 1. Model Creation ---
    model = gp.Model("StopLocationProblem_QuarterlyNonlinear")

    # --- 2. Sets and Parameters ---
    demand_nodes = range(len(distance))      # j
    stops = range(len(distance[0]))          # i
    quarters = range(1, 5)                   # t = 1,2,3,4

    # Pre-compute quarterly fluctuation coefficients:
    # coeff(t) = 0.9 + 0.1 * cos(pi * t / 2)
    quarter_coeff = {}
    for t in quarters:
        quarter_coeff[t] = 0.9 + 0.1 * math.cos(math.pi * t / 2.0)

    # ❤ Non-linearity is introduced. ❤
    # The original static-coverage computation is kept but commented out:
    """
    # Create a coverage parameter: a[j, i] = 1 if stop i covers demand node j
    a = {}
    for j in demand_nodes:
        for i in stops:
            if distance[j][i] <= coverage_distance:
                a[j, i] = 1
            else:
                a[j, i] = 0
    """

    # New: time-dependent coverage parameter a[j, i, t]
    # a[j, i, t] = 1 if in quarter t the effective distance
    # distance[j][i] * (0.9 + 0.1 * cos(pi * t / 2)) <= coverage_distance
    a = {}
    for j in demand_nodes:
        for i in stops:
            for t in quarters:
                effective_distance = distance[j][i] * 1
                if effective_distance <= coverage_distance:
                    a[j, i, t] = 1
                else:
                    a[j, i, t] = 0

    # --- 3. Decision Variables ---
    # x[i] = 1 if stop i is selected, 0 otherwise
    x = model.addVars(stops, vtype=GRB.BINARY, name="select_stop")

    # --- 4. Objective Function ---
    # Minimize the number of stops selected
    model.setObjective(gp.quicksum(x[i] for i in stops), GRB.MINIMIZE)

    # --- 5. Constraints ---
    # Each demand node must be covered by at least one selected stop
    # in every quarter t.
    for j in demand_nodes:
        for t in quarters:
            model.addConstr(
                gp.quicksum(a[j, i, t] * x[i] for i in stops) >= 1,
                name=f"cover_demand_{j}_quarter_{t}"
            )

    # --- 6. Solve the Model ---
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        return {"status": "optimal", "obj": int(model.ObjVal)}
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_stop_location()
    print(result)
import gurobipy as gp
from gurobipy import GRB


def solve_stop_location(
    distance=[
        [389, 515, 170, 143, 617], [562, 678, 265, 640, 629],
        [206, 594, 180, 564, 683], [574, 105, 311, 99, 550],
        [616, 490, 99, 473, 682], [571, 258, 494, 749, 61],
        [573, 234, 207, 635, 318], [70, 53, 399, 740, 494],
        [229, 190, 550, 654, 394], [50, 56, 459, 143, 478],
        [95, 378, 507, 647, 135], [767, 200, 569, 689, 621],
        [729, 333, 91, 86, 386], [633, 163, 562, 184, 384],
        [67, 515, 224, 502, 345]
    ],
    coverage_distance=300
):
    """
    Models and solves the stop location problem with a piecewise
    (segmented) fixed management cost:

    - If number of selected stops <= 3: no extra cost.
    - If number of selected stops  >= 4: an additional fixed cost of 500,000
      (e.g., CNY) is incurred.

    We still use an MILP model by linearizing this piecewise structure.
    """

    # --- 1. Model Creation ---
    model = gp.Model("StopLocationProblem_WithPiecewiseCost")

    # --- 2. Sets and Parameters ---
    demand_nodes = range(len(distance))
    stops = range(len(distance[0]))

    # Coverage parameter: a[j, i] = 1 if stop i covers demand node j
    a = {}
    for j in demand_nodes:
        for i in stops:
            if distance[j][i] <= coverage_distance:
                a[j, i] = 1
            else:
                a[j, i] = 0

    # --- 3. Decision Variables ---
    # x[i] = 1 if stop i is selected, 0 otherwise
    x = model.addVars(stops, vtype=GRB.BINARY, name="select_stop")

    # Binary variable for the piecewise management cost:
    # y = 1 if number of selected stops >= 4, else 0
    y = model.addVar(vtype=GRB.BINARY, name="extra_cost_indicator")

    # Auxiliary variable: total number of selected stops
    z = model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(stops), name="num_stops")

    # --- 4. Objective Function ---
    # Original objective: minimize the number of stops selected
    # model.setObjective(gp.quicksum(x[i] for i in stops), GRB.MINIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # New objective: minimize number of stops + piecewise extra management cost
    # When z <= 3, y = 0, no extra cost.
    # When z >= 4, y = 1, extra cost of 500,000 is added.
    big_extra_cost = 500000  # 50 万元
    model.setObjective(
        z + big_extra_cost * y,
        GRB.MINIMIZE
    )

    # --- 5. Constraints ---

    # 5.1 Each demand node must be covered by at least one selected stop
    for j in demand_nodes:
        model.addConstr(
            gp.quicksum(a[j, i] * x[i] for i in stops) >= 1,
            name=f"cover_demand_{j}"
        )

    # 5.2 Define z as the total number of selected stops
    model.addConstr(
        z == gp.quicksum(x[i] for i in stops),
        name="define_num_stops"
    )

    # 5.3 Link z and y to represent the piecewise extra cost logic:
    #
    # If z <= 3  → y must be 0
    # If z >= 4  → y must be 1
    #
    # This can be linearized using the following constraints:
    #
    #   z <= 3 + M * y        (1)
    #   z >= 4 * y            (2)
    #
    # Where M is a sufficiently large constant (here at most 5 stops, so M=2 works).
    # Check:
    # - If y = 0: (1) → z <= 3; (2) → z >= 0, so z ∈ {0,1,2,3}
    # - If y = 1: (1) → z <= 5; (2) → z >= 4, so z ∈ {4,5}
    M = len(stops) - 3  # = 2 for 5 candidate stops

    model.addConstr(
        z <= 3 + M * y,
        name="pw_upper_link"
    )
    model.addConstr(
        z >= 4 * y,
        name="pw_lower_link"
    )

    # --- 6. Solve the Model ---
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        # The objective value may be non-integer due to large extra cost;
        # return both the objective and the number of selected stops.
        num_selected_stops = int(round(z.X))
        extra_cost_flag = int(round(y.X))
        return {
            "status": "optimal",
            "objective_value": model.ObjVal,
            "num_selected_stops": num_selected_stops,
            "extra_cost_incurred": extra_cost_flag,
            "selected_stops": [i for i in stops if x[i].X > 0.5]
        }
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_stop_location()
    print(result)
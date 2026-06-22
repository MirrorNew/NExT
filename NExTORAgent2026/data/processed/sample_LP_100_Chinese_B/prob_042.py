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
    Models and solves the stop location problem with an additional
    non-linear type requirement:
    Among all selected stops, the ratio
        (# of stops that actually cover ≥1 demand point)
        / (# of selected stops)
    must be at least 0.8.
    """

    # --- 1. Model Creation ---
    model = gp.Model("StopLocationProblem")

    # --- 2. Sets and Parameters ---
    demand_nodes = range(len(distance))     # 15 demand nodes
    stops = range(len(distance[0]))         # 5 candidate stops

    # Coverage indicator: a[j, i] = 1 if stop i can cover demand j (within distance)
    a = {}
    for j in demand_nodes:
        for i in stops:
            if distance[j][i] <= coverage_distance:
                a[j, i] = 1
            else:
                a[j, i] = 0

    # --- 3. Decision Variables ---
    # x[i] = 1 if stop i is selected
    x = model.addVars(stops, vtype=GRB.BINARY, name="select_stop")

    # y[i] = 1 if stop i undertakes actual coverage (covers at least one demand)
    y = model.addVars(stops, vtype=GRB.BINARY, name="active_stop")

    # z[j, i] = 1 if demand j is actually assigned to (covered by) stop i
    z = model.addVars(demand_nodes, stops, vtype=GRB.BINARY, name="assign")

    # --- 4. Objective Function ---
    # Minimize number of selected stops
    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(gp.quicksum(x[i] for i in stops), GRB.MINIMIZE)
    # Keep same objective (linear), but the model will become non-linear due to
    # the coverage ratio constraint added later.
    model.setObjective(gp.quicksum(x[i] for i in stops), GRB.MINIMIZE)

    # --- 5. Constraints ---

    # 5.1 Each demand node must be covered by at least one selected stop
    #     (through assignment variable z)
    for j in demand_nodes:
        model.addConstr(
            gp.quicksum(z[j, i] for i in stops) >= 1,
            name=f"cover_demand_{j}"
        )

    # 5.2 Assignment only allowed to feasible stops (within coverage distance)
    for j in demand_nodes:
        for i in stops:
            model.addConstr(
                z[j, i] <= a[j, i],
                name=f"feasible_assign_{j}_{i}"
            )

    # 5.3 Assignment only allowed if stop is selected
    for j in demand_nodes:
        for i in stops:
            model.addConstr(
                z[j, i] <= x[i],
                name=f"assign_to_selected_{j}_{i}"
            )

    # 5.4 Define active stops y[i]: y[i] = 1 if stop i covers at least one demand
    #     Linearization: y[i] >= z[j, i] for all j
    for i in stops:
        for j in demand_nodes:
            model.addConstr(
                y[i] >= z[j, i],
                name=f"active_def_{i}_{j}"
            )

    # 5.5 Non-linear coverage ratio constraint:
    #     (#active stops) / (#selected stops) >= 0.8
    #     sum_i y[i] / sum_i x[i] >= 0.8
    #     This is a fractional (non-linear) constraint.
    # ❤ Non-linearity is introduced. ❤
    model.addConstr(
        gp.quicksum(y[i] for i in stops)  >= 0.8 * (gp.quicksum(x[i] for i in stops) + 1e-6),
        name="coverage_ratio_nonlinear"
    )

    # --- 6. Solve the Model ---
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        selected = [i for i in stops if x[i].X > 0.5]
        active = [i for i in stops if y[i].X > 0.5]
        return {
            "status": "optimal",
            "obj": int(round(model.ObjVal)),
            "selected_stops": selected,
            "active_stops": active,
            "num_selected": len(selected),
            "num_active": len(active),
            "ratio_active_selected":
                (len(active) / len(selected)) if len(selected) > 0 else None
        }
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_stop_location()
    print(result)
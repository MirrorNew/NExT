import gurobipy as gp
from gurobipy import GRB


def solve_transportation_problem(
    industry_storage=[60, 30, 45],
    shop_demand=[15, 35, 20, 5, 40],
    transportation_cost=[
        [4, 9, 2, 6, 5],
        [2, 6, 1, 7, 9],
        [2, 4, 9, 8, 3]
    ],
    fixed_cost=1000,
    threshold=50
):
    """
    Models and solves the transportation problem with additional
    fixed operating cost for each factory if its total shipment
    exceeds a given threshold.

    If total shipment from factory i > threshold, pay fixed_cost;
    otherwise, pay 0.
    """
    # --- 1. Model Creation ---
    model = gp.Model("TransportationProblem_With_FixedCost")

    # --- 2. Sets and Parameters ---
    industries = range(len(industry_storage))
    shops = range(len(shop_demand))

    # --- 3. Decision Variables ---

    # t[i, j] = amount transported from industry i to shop j
    t = model.addVars(industries, shops, vtype=GRB.INTEGER, name="transport")

    # Binary variable: y[i] = 1 if factory i's total shipment exceeds threshold,
    # otherwise 0. This will trigger the fixed cost.
    y = model.addVars(industries, vtype=GRB.BINARY, name="use_factory")

    # --- 4. Objective Function ---
    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(
    #     gp.quicksum(transportation_cost[i][j] * t[i, j] for i in industries for j in shops),
    #     GRB.MINIMIZE
    # )
    # New objective: transportation cost + fixed operating cost of factories
    model.setObjective(
        gp.quicksum(transportation_cost[i][j] * t[i, j]
                    for i in industries for j in shops)
        + gp.quicksum(fixed_cost * y[i] for i in industries),
        GRB.MINIMIZE
    )

    # --- 5. Constraints ---

    # Supply constraint for each industry
    for i in industries:
        model.addConstr(
            gp.quicksum(t[i, j] for j in shops) <= industry_storage[i],
            name=f"supply_{i}"
        )

    # Demand constraint for each shop
    for j in shops:
        model.addConstr(
            gp.quicksum(t[i, j] for i in industries) == shop_demand[j],
            name=f"demand_{j}"
        )

    # --- 5.1 Non-linear condition (modeled via MILP) ---
    # If total shipment from factory i exceeds 'threshold', pay fixed_cost.
    # Use a big-M style linearization:
    #
    # Let S_i = sum_j t[i, j]
    # We want:
    #   y_i = 0  ->  S_i <= threshold
    #   y_i = 1  ->  S_i > threshold
    #
    # We enforce:
    #   1) S_i <= threshold + M * y_i
    #   2) S_i >= threshold + 1 - M * (1 - y_i)
    #
    # Choose M as the maximum possible shipment from factory i,
    # which is its storage capacity industry_storage[i].
    for i in industries:
        S_i = gp.quicksum(t[i, j] for j in shops)
        M_i = industry_storage[i]

        # When y[i] = 0, this implies S_i <= threshold
        model.addConstr(
            S_i <= threshold + M_i * y[i],
            name=f"threshold_upper_{i}"
        )

        # When y[i] = 1, this implies S_i >= threshold + 1
        model.addConstr(
            S_i >= threshold + 1 - M_i * (1 - y[i]),
            name=f"threshold_lower_{i}"
        )

    # --- 6. Solve the Model ---
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        # Optionally, we can also return shipment plan and which factories pay fixed cost
        transport_plan = {
            (i, j): int(t[i, j].X) for i in industries for j in shops if t[i, j].X > 1e-6
        }
        used_factories = {i: int(y[i].X) for i in industries}
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "transport_plan": transport_plan,
            "used_factories": used_factories,
        }
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_transportation_problem()
    print(result)
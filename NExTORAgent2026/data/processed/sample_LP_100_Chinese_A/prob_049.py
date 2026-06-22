import gurobipy as gp
from gurobipy import GRB


def solve_transportation_problem(
):
    industry_storage = [60, 30, 45]
    shop_demand = [15, 35, 20, 5, 40]
    transportation_cost = [
        [4, 9, 2, 6, 5],
        [2, 6, 1, 7, 9],
        [2, 4, 9, 8, 3]
    ]
    """
    Models and solves the transportation problem with one nonlinear cost term:
    From factory 0 to shop 0 the total cost is (4 * x_0,0)^1.2 instead of 4 * x_0,0.
    All other routes keep linear costs.
    """
    # --- 1. Model Creation ---
    model = gp.Model("TransportationProblem_Nonlinear")

    # --- 2. Sets and Parameters ---
    industries = range(len(industry_storage))
    shops = range(len(shop_demand))

    # --- 3. Decision Variables ---
    # t[i, j] = amount transported from industry i to shop j
    t = model.addVars(industries, shops, vtype=GRB.INTEGER, name="transport")

    # --- 4. Objective Function ---
    # Minimize total transportation cost
    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(
    #     gp.quicksum(transportation_cost[i][j] * t[i, j] for i in industries for j in shops),
    #     GRB.MINIMIZE
    # )

    # Build the new objective: keep all linear terms except (0,0),
    # and replace the (0,0) term by (4 * t[0,0])^1.2
    linear_terms = gp.quicksum(
        transportation_cost[i][j] * t[i, j]
        for i in industries for j in shops
        if not (i == 0 and j == 0)
    )

    Y = model.addVar()
    model.addConstr(Y == 1 * t[0, 0])
    nonlinear_term = model.addVar()
    model.addGenConstrPow(Y, nonlinear_term, 1.2)
    model.setObjective(linear_terms + nonlinear_term, GRB.MINIMIZE)

    # --- 5. Constraints ---
    # Constraint 1: Supply constraint for each industry
    for i in industries:
        model.addConstr(
            gp.quicksum(t[i, j] for j in shops) <= industry_storage[i],
            name=f"supply_{i}"
        )

    # Constraint 2: Demand constraint for each shop
    for j in shops:
        model.addConstr(
            gp.quicksum(t[i, j] for i in industries) == shop_demand[j],
            name=f"demand_{j}"
        )

    # --- 6. Solve the Model ---
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "solution": {(i, j): t[i, j].X for i in industries for j in shops}
        }
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_transportation_problem()
    print(result)
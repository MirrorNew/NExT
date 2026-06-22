import gurobipy as gp
from gurobipy import GRB


def solve_transportation_problem(
    industry_storage=[60, 30, 45],
    shop_demand=[15, 35, 20, 5, 40],
    transportation_cost=[
        [4, 9, 2, 6, 5],
        [2, 6, 1, 7, 9],
        [2, 4, 9, 8, 3]
    ]
):
    """
    Models and solves the transportation problem with an added
    non-linear-type (ratio) balance condition:

    Total shipped from plant 1 / (total shipped from plants 2 + 3) <= 4/3
    """

    # --- 1. Model Creation ---
    model = gp.Model("TransportationProblem_With_Ratio")

    # --- 2. Sets and Parameters ---
    industries = range(len(industry_storage))
    shops = range(len(shop_demand))

    # --- 3. Decision Variables ---
    # t[i, j] = amount transported from industry i to shop j
    t = model.addVars(industries, shops, vtype=GRB.INTEGER, name="transport")

    # --- 4. Objective Function ---
    # Minimize total transportation cost
    model.setObjective(
        gp.quicksum(transportation_cost[i][j] * t[i, j] for i in industries for j in shops),
        GRB.MINIMIZE
    )

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

    # --- 6. Non-linear type constraint (ratio) ---
    # Let:
    #   P1 = total shipped from plant 1 (index 0)
    #   P23 = total shipped from plant 2 + 3 (indices 1 and 2)
    # Requirement: P1 / P23 <= 4/3
    # This is a ratio form, which is the source of non-linearity conceptually.
    # It can be linearized as: 3 * P1 <= 4 * P23

    P1 = gp.quicksum(t[0, j] for j in shops)
    P23 = gp.quicksum(t[1, j] + t[2, j] for j in shops)

    # ❤ Non-linearity is introduced. ❤
    # Original linear constraints did not contain this ratio relationship.
    model.addConstr(3 * P1 <= 4 * P23, name="ratio_balance_1_vs_23")

    # --- 7. Solve the Model ---
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    model.optimize()

    # --- 8. Return Results ---
    if model.status == GRB.OPTIMAL:
        # Collect solution details as well, in case they are needed.
        solution = {
            "status": "optimal",
            "obj": model.ObjVal,
            "transport_plan": {
                (i, j): t[i, j].X for i in industries for j in shops
            }
        }
        return solution
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_transportation_problem()
    print(result)
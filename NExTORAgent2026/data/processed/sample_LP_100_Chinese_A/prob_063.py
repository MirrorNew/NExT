def optimize_desktops(
    sales_limit=200,
    budget_limit=300000,
    premium_cost=2000,
    regular_cost=1000,
    premium_profit=500,
    regular_profit=300,
    power_exponent=1.002
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Desktop_Production_Optimization_Nonlinear")

    # Decision variables: number of premium and regular desktops
    x = model.addVar(vtype=GRB.INTEGER, name="Premium_Desktops", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="Regular_Desktops", lb=0)

    # Set objective: maximize profit
    model.setObjective(premium_profit * x + regular_profit * y, GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(x + y <= sales_limit, "Sales_Limit")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(premium_cost * x + regular_cost * y <= budget_limit, "Budget_Limit")
    # Nonlinear budget constraint: (total_cost)^1.2 <= (budget_limit)^1.2
    total_cost = premium_cost * x + regular_cost * y
    Y = model.addVar()
    model.addConstr(Y == total_cost, "Total Cost")
    YY = model.addVar()
    model.addGenConstrPow(Y, YY, power_exponent)
    model.addConstr(YY <= budget_limit, "Nonlinear_Budget_Limit")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the maximum profit, and also optimal production numbers for clarity
        return {
            "max_profit": model.objVal,
            "premium_desktops": int(x.X),
            "regular_desktops": int(y.X)
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_desktops()
    if result is not None:
        print(f"Maximum Profit: {result['max_profit']}")
        print(f"Optimal Premium Desktops: {result['premium_desktops']}")
        print(f"Optimal Regular Desktops: {result['regular_desktops']}")
    else:
        print("No feasible solution found.")
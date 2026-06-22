def optimize_desktops(sales_limit=200, budget_limit=300000, premium_cost=2000, regular_cost=1000,
                      premium_profit=500, regular_profit=300, extra_cost_threshold=150,
                      extra_fixed_cost=10000):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Desktop_Production_Optimization")

    # Decision variables: number of premium and regular desktops
    x = model.addVar(vtype=GRB.INTEGER, name="Premium_Desktops", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="Regular_Desktops", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # model.setObjective(premium_profit * x + regular_profit * y, GRB.MAXIMIZE)

    # Introduce a binary variable to model the fixed operating cost when production exceeds the threshold
    z = model.addVar(vtype=GRB.BINARY, name="Extra_Operating_Cost_Indicator")

    # Big-M constant for linking total production to the binary variable
    M = sales_limit  # Maximum possible total desktops, safe big-M here

    # Total production
    total_prod = x + y

    # If total_prod > extra_cost_threshold then z must be 1, else it can be 0.
    # Implement with big-M style constraints:
    # total_prod - extra_cost_threshold <= M * z
    model.addConstr(total_prod - extra_cost_threshold <= M * z, "Link_Upper_Threshold")
    # Also ensure that if z = 0, total_prod cannot exceed the threshold:
    model.addConstr(total_prod <= extra_cost_threshold + M * z, "Link_Lower_Threshold")

    # Set objective: maximize profit minus the extra fixed cost if z = 1
    model.setObjective(
        premium_profit * x + regular_profit * y - extra_fixed_cost * z,
        GRB.MAXIMIZE
    )

    # Add constraints
    model.addConstr(x + y <= sales_limit, "Sales_Limit")
    model.addConstr(premium_cost * x + regular_cost * y <= budget_limit, "Budget_Limit")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # You may also want to return x, y, and z values; here we return max profit only
        return model.objVal
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    max_profit = optimize_desktops()
    if max_profit is not None:
        print(f"Maximum Profit: {max_profit}")
    else:
        print("No feasible solution found.")
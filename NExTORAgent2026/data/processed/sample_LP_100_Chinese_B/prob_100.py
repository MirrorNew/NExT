def optimize_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("FlooringProduction")

    # Decision variables
    # x: square feet of laminate planks
    # y: square feet of carpets
    x = m.addVar(name="x", lb=15000, vtype=GRB.CONTINUOUS)
    y = m.addVar(name="y", lb=5000, vtype=GRB.CONTINUOUS)

    # Set the objective: maximize profit
    profit = 2.1 * x + 3.3 * y
    m.setObjective(profit, GRB.MAXIMIZE)

    # Add the shipping constraint
    m.addConstr(x + y >= 50000, name="shipping_constraint")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(x <= 40000, name="laminate_capacity")
    # m.addConstr(y <= 20000, name="carpet_capacity")
    # New nonlinear ratio constraint: x / y <= 2  ⇔  x <= 2 * y (non-linear form x / y)
    # Here we explicitly introduce the division to reflect the non-linear change in description.
    ratio_expr = x
    m.addConstr(ratio_expr <= 2 *y, name="ratio_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        return None


# Example usage
if __name__ == "__main__":
    max_profit = optimize_production()
    if max_profit is not None:
        print(f"Maximum Profit: ${max_profit}")
    else:
        print("No feasible solution found.")
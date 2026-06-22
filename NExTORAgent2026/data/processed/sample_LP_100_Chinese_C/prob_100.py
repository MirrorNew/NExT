def optimize_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("FlooringProduction")

    # Decision variables
    # x: square feet of laminate planks
    # y: square feet of carpets
    x = m.addVar(name="x", lb=15000, ub=40000, vtype=GRB.CONTINUOUS)
    y = m.addVar(name="y", lb=5000, ub=20000, vtype=GRB.CONTINUOUS)

    # ❤ Non-linearity is introduced. ❤
    # Binary variable to capture whether laminate production exceeds 30,000 sq ft
    z = m.addVar(name="z", vtype=GRB.BINARY)

    # Set the objective: maximize profit
    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (now commented out):
    # profit = 2.1 * x + 3.3 * y
    # m.setObjective(profit, GRB.MAXIMIZE)

    # New profit expression with a fixed cost of 8,000 if x > 30,000
    # Profit = 2.1 * x + 3.3 * y - 8000 * z
    profit = 2.1 * x + 3.3 * y - 8000 * z
    m.setObjective(profit, GRB.MAXIMIZE)

    # Add the shipping constraint
    m.addConstr(x + y >= 50000, name="shipping_constraint")

    # ❤ Non-linearity is introduced. ❤
    # Logic linking z and x for the step fixed cost:
    # If x <= 30,000 => z = 0 (no extra fixed cost)
    # If x > 30,000  => z = 1 (extra fixed cost 8,000 incurred)
    #
    # Using big-M formulation with existing bounds 15,000 <= x <= 40,000:
    # 1) x <= 30000 + (40000 - 30000) * z  = 30000 + 10000*z
    #    - If z = 0 => x <= 30000
    #    - If z = 1 => x <= 40000 (original upper bound)
    #
    # 2) x >= 30000 + 1 - (30000 - 15000) * (1 - z)
    #    = 30001 - 15000*(1 - z)
    #    - If z = 0 => x >= 30001 - 15000 = 15001 (non‑binding; within original lb)
    #    - If z = 1 => x >= 30001  (forces x strictly above 30000 when z = 1)
    m.addConstr(x <= 30000 + (40000 - 30000) * z, name="laminate_fixedcost_ub")
    m.addConstr(x >= 30001 - (30000 - 15000) * (1 - z), name="laminate_fixedcost_lb")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Optionally, you might also want to return x, y, and z, but
        # the original function only returns the objective value.
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
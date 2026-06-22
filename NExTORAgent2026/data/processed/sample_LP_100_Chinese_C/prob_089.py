def optimize_grilled_cheese():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("GrilledCheeseOptimization")
    
    # Decision variables: number of light and heavy sandwiches
    x = m.addVar(name="light_sandwiches", vtype=GRB.INTEGER, lb=0)
    y = m.addVar(name="heavy_sandwiches", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(10 * x + 15 * y, GRB.MINIMIZE)
    #
    # Non-linear feature: piecewise production time for heavy sandwiches
    # If y <= 40: all heavy sandwiches take 15 minutes each
    # If y > 40 : first 40 take 15 minutes, from 41st on take 20 minutes
    #
    # We linearize this with an extra integer variable:
    #   y1 = number of heavy sandwiches produced at 15 min each (capped at 40)
    #   y2 = number of heavy sandwiches produced at 20 min each (beyond 40)
    # such that:
    #   y = y1 + y2
    #   0 <= y1 <= 40
    #   y2 >= 0
    # This correctly represents:
    #   - if total y <= 40, optimal solution will set y1 = y, y2 = 0
    #   - if total y > 40, y1 will be capped at 40, remaining go to y2
    y1 = m.addVar(name="heavy_15min", vtype=GRB.INTEGER, lb=0, ub=40)
    y2 = m.addVar(name="heavy_20min", vtype=GRB.INTEGER, lb=0)

    # Link heavy-sandwich decomposition
    m.addConstr(y == y1 + y2, name="heavy_split")

    # Set the objective: minimize total production time with piecewise cost
    # Light: 10 min each
    # Heavy: y1 at 15 min, y2 at 20 min
    m.setObjective(10 * x + 15 * y1 + 20 * y2, GRB.MINIMIZE)
    
    # Add resource constraints
    m.addConstr(2 * x + 3 * y <= 300, name="bread_constraint")
    m.addConstr(3 * x + 5 * y <= 500, name="cheese_constraint")
    
    # Add demand ratio constraint
    m.addConstr(y >= 3 * x, name="heavy_light_ratio")
    
    # Optimize the model
    m.optimize()
    
    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total production time
        return m.objVal
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    min_time = optimize_grilled_cheese()
    if min_time is not None:
        print(f"Minimum Total Production Time: {min_time}")
    else:
        print("No feasible solution found.")
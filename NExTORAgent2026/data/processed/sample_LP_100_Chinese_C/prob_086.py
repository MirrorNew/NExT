from gurobipy import Model, GRB

def optimize_wraps_and_platters():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("FastFoodProduction")

    # Decision variables: number of wraps (x) and platters (y)
    x = m.addVar(vtype=GRB.INTEGER, name="wraps", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="platters", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(10 * x + 8 * y, GRB.MINIMIZE)

    # Introduce an auxiliary variable for wraps beyond 400 (non-linear time rule)
    x_extra = m.addVar(vtype=GRB.INTEGER, name="wraps_extra", lb=0)

    # Link x_extra with total wraps x: x_extra = max(0, x - 400)
    m.addConstr(x_extra >= x - 400, name="extra_wraps_lb")
    m.addConstr(x_extra <= x, name="extra_wraps_ub1")
    m.addConstr(x_extra <= 400, name="extra_wraps_ub2")

    # New objective:
    # First 400 wraps (or fewer): 10 minutes each
    # Wraps beyond 400: 12 minutes each
    # Total wrap time = 10 * x + 2 * x_extra
    # Platters: 8 minutes each
    m.setObjective(10 * x + 2 * x_extra + 8 * y, GRB.MINIMIZE)

    # Add resource constraints
    m.addConstr(5 * x + 7 * y >= 3000, name="meat_constraint")
    m.addConstr(3 * x + 5 * y >= 2500, name="rice_constraint")

    # Add production ratio constraint
    m.addConstr(x - 3 * y >= 0, name="wraps_vs_platters")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total production time and decision variables
        return {
            "min_time": m.objVal,
            "wraps": x.X,
            "platters": y.X
        }
    else:
        # No feasible solution found
        return None

# Example usage
if __name__ == "__main__":
    result = optimize_wraps_and_platters()
    if result is not None:
        print(f"Minimum Total Production Time: {result['min_time']}")
        print(f"Optimal number of wraps: {result['wraps']}")
        print(f"Optimal number of platters: {result['platters']}")
    else:
        print("No feasible solution found.")
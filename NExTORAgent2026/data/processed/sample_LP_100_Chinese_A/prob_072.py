from gurobipy import Model, GRB

def optimize_ski_lifts():
    # Create a new model
    m = Model("SkiLiftsOptimization_Nonlinear")

    # Decision variables
    # x: number of densely-seated lifts
    # y: number of loosely-seated lifts
    # z: number of tourists entering the system per minute (queue arrivals)
    x = m.addVar(vtype=GRB.INTEGER, name="x", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="y", lb=5)
    z = m.addVar(vtype=GRB.CONTINUOUS, name="z", lb=1000)

    # Set the objective: minimize total number of lifts
    m.setObjective(x + y, GRB.MINIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Original linear capacity constraint:
    # m.addConstr(45 * x + 20 * y >= 1000, name="capacity")

    # Non-linear capacity constraint with synergy term:
    # total capacity = 45*x + 20*y + 5*x*y*z  >=  1000
    Y = m.addVar()
    m.addConstr(Y == x * y)
    m.addConstr(45 * x + 20 * y + 1e-11 * Y * z >= 1000, name="capacity_nonlinear")

    # Electricity constraint (unchanged)
    m.addConstr(30 * x + 22 * y <= 940, name="electricity")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal decision variables and objective value
        return {
            "x": x.X,
            "y": y.X,
            "z": z.X,
            "min_total_lifts": m.objVal
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_ski_lifts()
    if result is not None:
        print(f"Optimal number of densely-seated lifts (x): {result['x']}")
        print(f"Optimal number of loosely-seated lifts (y): {result['y']}")
        print(f"Optimal number of tourists entering per minute (z): {result['z']}")
        print(f"Minimum Total Lifts: {result['min_total_lifts']}")
    else:
        print("No feasible solution found.")
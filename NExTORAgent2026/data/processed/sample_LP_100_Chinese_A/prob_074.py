import math


def optimize_saws():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("woodshop_saws_nonlinear")

    # Decision variables: number of each saw type
    x = m.addVar(vtype=GRB.INTEGER, name="miter_saws", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="circular_saws", lb=0)

    # Set the objective: minimize total number of saws
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add constraints
    # Wood cutting constraint
    m.addConstr(50 * x + 70 * y >= 1500, name="wood_cutting")

    # Sawdust production / effective pollution constraint
    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(60 * x + 100 * y <= 2000, name="sawdust_limit")
    # Effective pollution: (60*x + 100*y) * 1.3^(x + y) <= 2000
    # Implemented via Gurobi's general function definition and constraint

    # Define exponent base and maximum reasonable bound for x + y
    base = 1.2

    # Create an auxiliary variable for total saws to be used inside the nonlinear term
    t = m.addVar(vtype=GRB.INTEGER, name="total_saws")

    # Link t with x and y
    m.addConstr(t == math.log(base) * (x), name="total_saws_def")

    # Auxiliary variable for effective pollution
    p = m.addVar(vtype=GRB.CONTINUOUS, name="effective_pollution")

    # Register a general function for base^t  (exponential in integer t)
    # f(t) = base^t
    m.addGenConstrExp(t, p)

    # Now p = exp(log(base) * t) = base^t.
    # We need (60*x + 100*y) * base^(x + y) <= 2000
    # So define another variable for (60*x + 100*y)
    linear_sawdust = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="linear_sawdust")

    m.addConstr(linear_sawdust == 60 * x + 100 * y, name="linear_sawdust_def")

    # Variable for effective pollution = linear_sawdust * p
    eff = m.addVar(vtype=GRB.CONTINUOUS, name="effective_pollution_total")

    # Use a general bilinear constraint: eff = linear_sawdust * p
    m.addConstr(eff == linear_sawdust + p )

    # Effective pollution limit
    m.addConstr(eff <= 2201, name="effective_pollution_limit")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of saws and the breakdown
        return {
            "total_saws": m.objVal,
            "miter_saws": x.X,
            "circular_saws": eff.X
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_saws()
    if result is not None:
        print(f"Minimum Total Number of Saws: {result['total_saws']}")
        print(f"  Miter Saws (x): {result['miter_saws']}")
        print(f"  Circular Saws (y): {result['circular_saws']}")
    else:
        print("No feasible solution found.")
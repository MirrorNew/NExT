import math

from gurobipy import Model, GRB


def optimize_wraps_and_platters():
    # Create a new model
    m = Model("FastFoodProduction_Nonlinear")

    # Decision variables: number of wraps (x) and platters (y)
    x = m.addVar(vtype=GRB.INTEGER, name="wraps", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="platters", lb=0)

    # Set objective: minimize total production time
    m.setObjective(10 * x + 8 * y, GRB.MINIMIZE)

    # Add resource constraints
    m.addConstr(5 * x + 7 * y >= 3000, name="meat_constraint")
    m.addConstr(3 * x + 5 * y >= 2500, name="rice_constraint")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(x - 3 * y >= 0, name="wraps_vs_platters")

    # Non-linear production ratio constraint:
    # x >= 3 * (1.3 ** y)
    # Gurobi supports general constraints through genconstrs; we use a general
    # power expression to represent 1.3 ** y, then link it with x.
    #
    # Note: This creates a non-linear (mixed-integer nonlinear) model.
    #       Requires a Gurobi license with NLP capabilities.

    # Auxiliary variable for 1.3 ** y
    z = m.addVar(vtype=GRB.CONTINUOUS, name="exp_term", lb=0.0)
    lnz = m.addVar(vtype=GRB.CONTINUOUS, name="z")

    # Add general power constraint: z = (1.3) ^ y
    m.addConstr(lnz == y * math.log(1.3) )
    m.addGenConstrLog(z,lnz)

    # Now enforce x >= 3 * z
    m.addConstr(x >= 3 * z, name="wraps_vs_platters_nonlinear")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total production time and solution details
        return m.objVal, int(x.X), int(y.X)
    else:
        # No feasible solution found
        return None, None, None


# Example usage
if __name__ == "__main__":
    min_time, wraps_opt, platters_opt = optimize_wraps_and_platters()
    if min_time is not None:
        print(f"Minimum Total Production Time: {min_time}")
        print(f"Optimal number of wraps: {wraps_opt}")
        print(f"Optimal number of platters: {platters_opt}")
    else:
        print("No feasible solution found.")
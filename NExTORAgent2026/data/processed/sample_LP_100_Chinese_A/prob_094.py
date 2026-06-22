import math


def optimize_transportation():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("TransportationOptimization")

    # Decision variables
    b = m.addVar(vtype=GRB.INTEGER, name="bikes", lb=0)
    c = m.addVar(vtype=GRB.INTEGER, name="cars", lb=0)

    # Set objective: minimize number of bikes
    m.setObjective(b, GRB.MINIMIZE)

    # ---------------- Capacity constraint with non-linear effect ----------------
    # Original linear capacity constraint (commented out):
    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(3 * b + 5 * c >= 500, name="capacity_constraint")

    # ❤ Non-linearity is introduced. ❤
    # With exponential growth in required people: required_people = 500 * 1.3^c
    # Capacity must be at least this new required_people
    # 3*b + 5*c >= 500 * 1.3^c

    # Note: 1.3^c = exp(c * ln(1.3)), so we linearize via another variable
    ln_1_3 = math.log(1.0005)               # precomputed ln(1.3)

    # Create an auxiliary continuous variable z = c * ln(1.3)
    z = m.addVar(vtype=GRB.CONTINUOUS, name="z")

    # z = ln(1.3) * c
    m.addConstr(z == ln_1_3 * c, name="z_def")

    # y = exp(z) = 1.3^c
    y = m.addVar(vtype=GRB.CONTINUOUS, name="growth_factor", lb=0)
    m.addGenConstrExp(z, y, name="exp_growth")

    # Now impose the non-linear capacity requirement using y = 1.3^c
    m.addConstr(3 * b + 5 * c >= 500 * y, name="capacity_constraint_nonlinear")

    # ---------------- Vehicle ratio constraint ----------------
    # Car count cannot exceed 40% of total vehicles:
    # c <= 0.4 * (b + c)  ->  0.6c <= 0.4b  ->  3c <= 2b
    m.addConstr(3 * c <= 2 * b, name="vehicle_ratio_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the minimal number of bikes and cars
        return int(b.X), int(c.X)
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_transportation()
    if result is not None:
        min_bikes, cars = result
        print(f"Minimum number of bikes: {min_bikes}")
        print(f"Number of cars: {cars}")
    else:
        print("No feasible solution found.")
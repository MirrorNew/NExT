def optimize_ski_lifts():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("SkiLiftsOptimization")

    # Decision variables
    # x: number of densely-seated lifts
    # y: number of loosely-seated lifts
    x = m.addVar(vtype=GRB.INTEGER, name="x", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="y", lb=5)

    # ❤ Non-linearity is introduced. ❤
    # Electricity constraint without peak system
    # m.addConstr(30 * x + 22 * y <= 940, name="electricity")

    # Binary variable to indicate whether the high-density lift count exceeds 15
    # z = 1 if x >= 16, and z = 0 if x <= 15 (enforced by big-M constraints)
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Big-M value: an upper bound on possible x (choose safely large but not excessive)
    M = 1000

    # Link x and z to model the condition "x >= 16  =>  z = 1", "x <= 15 => z = 0"
    # When z = 0: x <= 15
    m.addConstr(x <= 15 + M * z, name="link_upper_x_z")
    # When z = 1: x >= 16
    m.addConstr(x >= 16 * z, name="link_lower_x_z")

    # Electricity constraint with conditional extra 120 units if x >= 16:
    # 30*x + 22*y + 120*z <= 940
    m.addConstr(30 * x + 22 * y + 120 * z <= 940, name="electricity_piecewise")

    # Set the objective: minimize total number of lifts
    m.setObjective(x + y, GRB.MINIMIZE)

    # Capacity constraint
    m.addConstr(45 * x + 20 * y >= 1000, name="capacity")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of lifts and the composition
        return {
            "min_total_lifts": int(m.objVal),
            "x_high_density": int(x.X),
            "y_low_density": int(y.X),
            "z_peak_system": int(z.X),
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_ski_lifts()
    if result is not None:
        print(f"Minimum Total Lifts: {result['min_total_lifts']}")
        print(f"High-density lifts (x): {result['x_high_density']}")
        print(f"Low-density lifts (y): {result['y_low_density']}")
        print(
            f"Peak system active (z=1 means x>=16): {bool(result['z_peak_system'])}"
        )
    else:
        print("No feasible solution found.")
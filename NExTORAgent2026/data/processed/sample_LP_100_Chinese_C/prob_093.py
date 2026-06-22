def optimize_vehicles():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("airport_vehicles")

    # Decision variables: number of 4-wheelers and 3-wheelers
    x = m.addVar(vtype=GRB.INTEGER, name="x")  # 4-wheeler
    y = m.addVar(vtype=GRB.INTEGER, name="y")  # 3-wheeler

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective:
    # m.setObjective(x + y, GRB.MINIMIZE)
    # Replace with a (demonstratively) non-linear objective:
    # minimize total vehicles plus a quadratic penalty on 4-wheelers
    m.setObjective(x + y + 0.01 * x * x, GRB.MINIMIZE)

    # Add constraints
    # Luggage capacity constraint
    m.addConstr(60 * x + 40 * y >= 1000, "luggage_constraint")

    # ❤ Non-linearity is introduced. ❤
    # Original linear pollution constraint:
    # m.addConstr(30 * x + 15 * y <= 430, "pollution_constraint")
    # Replace with a non-linear (quadratic) pollution constraint:
    # base pollution from vehicles + additional pollution that grows with x
    # (this is just to introduce non-linearity; the logical “if x > 10 then +50”
    # part is a separate piece handled below with a binary variable and big-M
    # linearization)
    m.addQConstr(30 * x + 15 * y + 0.1 * x * x <= 430, "pollution_constraint_nl")

    # ---- Logical part: extra 50 units of pollution if x > 10 ----
    # This part models the “if x > 10 then +50 pollution” using a binary
    # variable and big-M, which is MILP (still linear in x, y, z).
    # Decision variable: z = 1 if x > 10, 0 otherwise
    z = m.addVar(vtype=GRB.BINARY, name="z")  # indicator of extra facilities

    # Big-M parameter large enough to cover any feasible x range
    M = 1000

    # If z = 0, force x <= 10; if z = 1, allow x > 10
    m.addConstr(x <= 10 + M * z, name="x_le_10_or_z1")
    # If z = 1, force x >= 11; if z = 0, no lower bound beyond non-negativity
    m.addConstr(x >= 11 * z, name="x_ge_11_if_z1")

    # Pollution accounting with extra 50 when z = 1.
    # Note: this is an additional linear pollution cap that uses z.
    # It coexists with the non-linear pollution constraint above.
    m.addConstr(30 * x + 15 * y + 50 * z <= 430, name="pollution_with_extra_50")

    # Non-negativity is implicit in variable definition

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return objective value and solution details for completeness
        return {
            "objective_value": m.objVal,
            "x_4wheelers": x.X,
            "y_3wheelers": y.X,
            "z_extra_facility": z.X,
        }
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_vehicles()
    if result is not None:
        print(f"Minimum Objective Value (with non-linearity): {result['objective_value']}")
        print(f"Number of 4-wheelers (x): {result['x_4wheelers']}")
        print(f"Number of 3-wheelers (y): {result['y_3wheelers']}")
        print(f"Extra facility built (z=1 means yes): {result['z_extra_facility']}")
    else:
        print("No feasible solution found.")
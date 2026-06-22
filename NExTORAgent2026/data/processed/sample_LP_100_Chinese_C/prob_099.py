def optimize_printers():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Printer_Optimization")

    # Decision variables: number of premium and regular printers
    x = m.addVar(vtype=GRB.INTEGER, name="Premium")
    y = m.addVar(vtype=GRB.INTEGER, name="Regular")

    # ❤ Non-linearity is introduced. ❤
    # Set objective: minimize total number of printers
    # m.setObjective(x + y, GRB.MINIMIZE)
    # Introduce a binary variable to capture the non-linear management rule
    z = m.addVar(vtype=GRB.BINARY, name="HighEndGE5")

    # Modified objective:
    # We still minimize the *physical* total number of printers x + y.
    # The non-linearity is represented by logical / indicator constraints instead.
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add constraints
    # Printing capacity constraint
    m.addConstr(30 * x + 20 * y >= 200, name="PrintCapacity")
    # Ink usage constraint
    m.addConstr(4 * x + 3 * y <= 35, name="InkLimit")
    # Preference constraint: y < x
    m.addConstr(y <= x - 1, name="Preference")

    # ❤ Non-linearity is introduced. ❤
    # Additional management rule with logical (non-linear in description) behavior:
    # If x >= 5 (high-end printers at least 5), then at least 1 extra regular printer is required.
    # We model this via a binary variable z and big-M constraints.

    M = 100  # A sufficiently large constant

    # Link z with the logical condition "x >= 5"
    # When z = 1, force x >= 5
    m.addConstr(x >= 5 * z, name="HighEndAtLeast5_if_z1")

    # When z = 0, force x <= 4 (so that x < 5)
    m.addConstr(x <= 4 + M * z, name="HighEndLessThan5_if_z0")
    # Explanation:
    #  - If z = 0, this becomes x <= 4, i.e., x < 5.
    #  - If z = 1, this becomes x <= 4 + M, which is non-binding for large M.

    # Extra-regular-printer rule:
    # If x >= 5 (i.e., z = 1), then total regular printers y must be at least baseline + 1.
    # Here baseline "y" already represents total regular printers, so we encode:
    #    z = 1  =>  y >= 1
    m.addConstr(y >= z, name="ExtraRegularIfHighEndGE5")

    # Non-negativity constraints are implicit in variable definitions (integer variables are >= 0)

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the total number of printers in the optimal solution
        total_printers = x.X + y.X
        print(f"Optimal Premium (High-end) Printers: {int(x.X)}")
        print(f"Optimal Regular Printers: {int(y.X)}")
        return total_printers
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    min_printers = optimize_printers()
    if min_printers is not None:
        print(f"Minimum Total Number of Printers: {int(min_printers)}")
    else:
        print("No feasible solution found.")
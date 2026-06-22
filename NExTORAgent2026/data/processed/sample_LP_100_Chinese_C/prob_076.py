def optimize_vitamins():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("VitaminIntakeMaximization")

    # Decision variables: number of orange and apple juice boxes
    x = m.addVar(name="orange_boxes", vtype=GRB.INTEGER, lb=3)
    y = m.addVar(name="apple_boxes", vtype=GRB.INTEGER, lb=0)

    # Binary variable to indicate whether total boxes exceed 25
    # z = 0  -> x + y <= 25
    # z = 1  -> x + y  > 25 (up to the implied maximum bound)
    z = m.addVar(name="total_boxes_over_25", vtype=GRB.BINARY)

    # Set the objective: maximize total vitamin D
    m.setObjective(10 * x + 12 * y, GRB.MAXIMIZE)

    # Add constraints
    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(8 * x + 6 * y <= 300, name="VitaminC_limit")

    # Piecewise / conditional vitamin C limit using a binary variable
    # Vitamin C limit is:
    #   <= 300 when x + y <= 25
    #   <= 260 when x + y  > 25

    # Big-M style linearization
    # Choose an upper bound for total boxes; from other constraints, y >= 3x and x >= 3
    # will keep x + y reasonably bounded. We pick a safe big M for demonstration.
    M_boxes = 100  # Big-M for total boxes
    M_C = 300      # Big-M for vitamin C slack between 300 and 260

    # Link z with total boxes: if z = 0, enforce x + y <= 25; if z = 1, allow up to 25 + M_boxes
    m.addConstr(x + y <= 25 + M_boxes * z, name="TotalBoxes_upper_branch")
    # Optional strengthening: when z = 1, force x + y >= 26 (strictly greater than 25 in integer context)
    m.addConstr(x + y >= 26 * z, name="TotalBoxes_lower_branch")

    # Vitamin C constraints:
    # When z = 0 (x + y <= 25): 8x + 6y <= 300
    # When z = 1 (x + y > 25):  8x + 6y <= 260
    #
    # Implement as:
    #   8x + 6y <= 300                      when z = 0
    #   8x + 6y <= 260 + M_C * (1 - z)      when z = 1
    #
    # Combined linear constraints:
    m.addConstr(8 * x + 6 * y <= 300, name="VitaminC_limit_300")
    m.addConstr(8 * x + 6 * y <= 260 + M_C * (1 - z), name="VitaminC_limit_260")

    # Preference constraint: at least 3 times as many apple as orange
    m.addConstr(y >= 3 * x, name="Preference")
    # Minimum orange boxes
    m.addConstr(x >= 3, name="Min_orange")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum vitamin D intake and chosen box counts for clarity
        return {
            "max_vitamin_d": m.objVal,
            "orange_boxes": x.X,
            "apple_boxes": y.X,
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_vitamins()
    if result is not None:
        print(f"Maximum Vitamin D Intake: {result['max_vitamin_d']}")
        print(f"Orange juice boxes: {result['orange_boxes']}")
        print(f"Apple juice boxes: {result['apple_boxes']}")
    else:
        print("No feasible solution found.")
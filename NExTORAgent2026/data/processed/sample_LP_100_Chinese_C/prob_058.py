def optimize_fertilizer_cost():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Fertilizer_Optimization")

    # Decision variables: amount of fertilizers C and Y
    x = m.addVar(name="C", lb=0)  # fertilizer C in kg
    y = m.addVar(name="Y", lb=0)  # fertilizer Y in kg

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)
    # Introduce a binary variable to capture the fixed cost when y > 2
    z = m.addVar(vtype=GRB.BINARY, name="Y_over_2_indicator")

    # Big-M value: an upper bound on y (can be chosen large enough; here 100 is safe)
    M = 100.0

    # Logical constraints to model "z = 1 if y > 2, else z = 0"
    # When z = 0  =>  y <= 2
    m.addConstr(y <= 2 + M * z, name="Y_upper_logic")
    # When z = 1  =>  y >= 2 + small_epsilon (e.g., 1e-6)
    m.addConstr(y >= 2 + 1e-6 * z, name="Y_lower_logic")

    # Objective with fixed cost: 2*C + 3*Y + 4*z
    m.setObjective(2 * x + 3 * y + 4 * z, GRB.MINIMIZE)

    # Add constraints
    m.addConstr(1.5 * x + 5 * y >= 5, name="NitrousOxide")
    m.addConstr(3 * x + y >= 8, name="VitaminMix")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        return None


# Example usage
if __name__ == "__main__":
    min_cost = optimize_fertilizer_cost()
    if min_cost is not None:
        print(f"Minimum Cost of Fertilizer Mixture: {min_cost}")
    else:
        print("No feasible solution found.")
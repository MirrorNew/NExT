def optimize_coins(gold_available=500, wires_available=300):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Coin_Plation_Optimization")

    # Decision variables: number of processes A and B
    x_A = model.addVar(vtype=GRB.INTEGER, name="Process_A")
    x_B = model.addVar(vtype=GRB.INTEGER, name="Process_B")

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(5 * x_A + 7 * x_B, GRB.MAXIMIZE)
    # Introduce a binary variable y_B to indicate whether Process B runs more than 50 times
    y_B = model.addVar(vtype=GRB.BINARY, name="B_over_50")

    # Objective: maximize total coins (still linear in variables)
    model.setObjective(5 * x_A + 7 * x_B, GRB.MAXIMIZE)

    # Add resource constraints
    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(3 * x_A + 5 * x_B <= gold_available, "GoldConstraint")
    # If x_B > 50, an extra fixed cost equivalent to 40 units of gold is consumed.
    # Linearized with binary y_B:
    #   - When y_B = 0  ⇒ x_B ≤ 50, no extra gold consumed.
    #   - When y_B = 1  ⇒ x_B can exceed 50, but gold available is reduced by 40.
    big_M = gold_available  # sufficiently large upper bound for x_B

    # Ensure that if y_B = 0 then x_B ≤ 50; if y_B = 1, x_B can be up to big_M
    model.addConstr(x_B <= 50 + big_M * y_B, "B_over_50_link")

    # Gold constraint with extra 40 units consumed when y_B = 1
    model.addConstr(3 * x_A + 5 * x_B + 40 * y_B <= gold_available, "GoldConstraint_with_FixedCost")

    # Wire constraint remains unchanged
    model.addConstr(2 * x_A + 3 * x_B <= wires_available, "WiresConstraint")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the maximum total number of coins
        return model.objVal
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    max_coins = optimize_coins()
    if max_coins is not None:
        print(f"Maximum Total Coins: {max_coins}")
    else:
        print("No feasible solution found.")
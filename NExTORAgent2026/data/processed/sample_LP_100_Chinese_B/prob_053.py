def optimize_coins(gold_available=500, wires_available=300):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Coin_Plation_Optimization")

    # Decision variables: number of processes A and B
    x_A = model.addVar(vtype=GRB.INTEGER, name="Process_A")
    x_B = model.addVar(vtype=GRB.INTEGER, name="Process_B")

    # Set the objective: maximize total coins
    model.setObjective(5 * x_A + 7 * x_B, GRB.MAXIMIZE)

    # Add resource constraints
    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(3 * x_A + 5 * x_B <= gold_available, "GoldConstraint")
    # model.addConstr(2 * x_A + 3 * x_B <= wires_available, "WiresConstraint")

    # Original linear resource constraints are still valid bounds:
    model.addConstr(3 * x_A + 5 * x_B <= gold_available, "GoldConstraint")
    model.addConstr(2 * x_A + 3 * x_B <= wires_available, "WiresConstraint")

    # ❤ Non-linearity is introduced. ❤
    # New nonlinear ratio constraint:
    # (total gold used) / (total wires used) <= 500 / 300
    # i.e., (3*x_A + 5*x_B) / (2*x_A + 3*x_B) <= 500/300
    # This introduces a nonlinear fractional constraint.
    total_gold = 3 * x_A + 5 * x_B
    total_wires = 2 * x_A + 3 * x_B
    model.addQConstr(
        total_gold * 300 <= 500 * total_wires,
        name="GoldWireRatioConstraint"
    )

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
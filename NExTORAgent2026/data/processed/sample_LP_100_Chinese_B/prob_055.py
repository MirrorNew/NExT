def optimize_meat_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Meat_Processing_Optimization")

    # Decision variables: number of batches of hams and pork ribs
    x = m.addVar(name="Hams", lb=0)
    y = m.addVar(name="Pork_Ribs", lb=0)

    # Set the objective: maximize profit
    m.setObjective(150 * x + 300 * y, GRB.MAXIMIZE)

    # Add constraints
    # Meat slicer constraint
    slicer_time = 4 * x + 2 * y
    m.addConstr(slicer_time <= 4000, name="Slicer_Time")

    # Meat packer constraint
    packer_time = 2.5 * x + 3.5 * y
    m.addConstr(packer_time <= 4000, name="Packer_Time")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(4 * x + 2 * y <= 4000, name="Slicer_Time")
    # m.addConstr(2.5 * x + 3.5 * y <= 4000, name="Packer_Time")

    # ❤ Non-linearity is introduced. ❤
    # Introduce the nonlinear utilization coordination constraint:
    # (total slicer time) / (total packer time) <= 1.1
    # i.e., (4x + 2y) / (2.5x + 3.5y) <= 1.1
    # This is a nonlinear (rational) constraint.
    m.addQConstr(slicer_time <= 1.1 * packer_time, name="Utilization_Ratio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal profit
        return m.objVal
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    max_profit = optimize_meat_production()
    if max_profit is not None:
        print(f"Maximum Profit: ${max_profit}")
    else:
        print("No feasible solution found.")
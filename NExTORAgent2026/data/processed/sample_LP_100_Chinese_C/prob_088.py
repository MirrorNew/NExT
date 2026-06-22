def optimize_appliances():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Appliance_Optimization")

    # Decision variables: number of refrigerators and stoves
    # Both are integers and non-negative
    x = m.addVar(vtype=GRB.INTEGER, name="Refrigerators", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="Stoves", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective is commented out:
    # m.setObjective(400 * x + 260 * y, GRB.MAXIMIZE)

    # Introduce binary variables for fixed-cost triggers
    # b_ref = 1 if x > 150 (extra cold-chain truck needed), else 0
    # b_stove = 1 if y > 180 (extra installer needed), else 0
    b_ref = m.addVar(vtype=GRB.BINARY, name="Extra_Truck")
    b_stove = m.addVar(vtype=GRB.BINARY, name="Extra_Installer")

    # Big-M style linking constraints to approximate the step condition:
    # If b_ref = 0 then x ≤ 150; if x ≥ 151 then b_ref must be 1.
    # If b_stove = 0 then y ≤ 180; if y ≥ 181 then b_stove must be 1.
    Mx = 10000  # a sufficiently large number for refrigerators
    My = 10000  # a sufficiently large number for stoves

    # x <= 150 + Mx * b_ref
    m.addConstr(x <= 150 + Mx * b_ref, "TruckUpperLink")
    # x >= 151 * b_ref
    m.addConstr(x >= 151 * b_ref, "TruckLowerLink")

    # y <= 180 + My * b_stove
    m.addConstr(y <= 180 + My * b_stove, "InstallerUpperLink")
    # y >= 181 * b_stove
    m.addConstr(y >= 181 * b_stove, "InstallerLowerLink")

    # New objective with fixed costs deducted when thresholds are exceeded
    # Profit = 400x + 260y - 5000*b_ref - 3000*b_stove
    m.setObjective(400 * x + 260 * y - 5000 * b_ref - 3000 * b_stove, GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(60 * x + 45 * y <= 20000, "MoverTime")
    m.addConstr(20 * x + 25 * y <= 13000, "SetupTime")

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
    max_profit = optimize_appliances()
    if max_profit is not None:
        print(f"Maximum Profit: ${max_profit}")
    else:
        print("No feasible solution found.")
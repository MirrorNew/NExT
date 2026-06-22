def optimize_car_jacks():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("CarJacksOptimization")

    # Decision variables
    # x: number of automatic electric jacks
    # y: number of gas-powered jacks
    x = m.addVar(vtype=GRB.INTEGER, name="electric_jacks")
    y = m.addVar(vtype=GRB.INTEGER, name="gas_jacks")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(6 * x <= 50, "electricity_limit")
    # ---- Introduce binary variable to model the piecewise electricity capacity ----
    # z = 0  => x <= 10, available electricity = 50
    # z = 1  => x >= 11, available electricity = 40 (because 10 units reserved for protection device)
    z = m.addVar(vtype=GRB.BINARY, name="protection_on")

    # Link z with x (big-M logic)
    # If z = 0, x <= 10
    # If z = 1, x can go up to 14 (original upper bound)
    m.addConstr(x <= 10 + 4 * z, name="electric_jacks_logic_upper")  # 10 + 4*1 = 14
    m.addConstr(x >= 11 * z, name="electric_jacks_logic_lower")     # if z=1, x >= 11; if z=0, x >= 0

    # Electricity capacity dependent on z:
    # z = 0 => 6x <= 50
    # z = 1 => 6x <= 40
    # This can be modeled compactly with a big-M style inequality:
    # 6x <= 40 + 10*(1 - z)
    m.addConstr(6 * x <= 40 + 10 * (1 - z), name="electricity_piecewise")

    # Set the objective: maximize total cars processed
    m.setObjective(5 * x + 4 * y, GRB.MAXIMIZE)

    # Add remaining constraints
    m.addConstr(x <= 14, "max_electric_jacks")
    m.addConstr(7 * y <= 80, "gas_limit")
    m.addConstr(x >= 0, "non_neg_electric")
    m.addConstr(y >= 0, "non_neg_gas")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum number of cars processed
        return m.objVal
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    max_cars = optimize_car_jacks()
    if max_cars is not None:
        print(f"Maximum Cars Processed: {max_cars}")
    else:
        print("No feasible solution found.")
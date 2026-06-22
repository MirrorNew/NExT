def optimize_shifts(total_shifts=40, energy_limit=230, min_orders=320, min_scooter_shifts=5):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Delivery_Shifts_Maximize_Tips")

    # Decision variables: number of bike and scooter shifts
    x = model.addVar(vtype=GRB.INTEGER, name="bike_shifts", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="scooter_shifts", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(50 * x + 43 * y, GRB.MAXIMIZE)

    # Introduce an auxiliary variable for scooter shifts beyond 20 (non-linear change)
    y_extra = model.addVar(vtype=GRB.INTEGER, name="scooter_shifts_extra", lb=0)

    # Link y_extra with y: y_extra = max(0, y - 20)
    model.addConstr(y_extra >= y - 20, "extra_scooter_lb")
    model.addConstr(y_extra <= y, "extra_scooter_ub1")
    model.addConstr(y_extra <= 20, "extra_scooter_ub2")

    # Objective: first 20 scooter shifts give 43 each, extra ones give 38 each
    # 50*x + 43*min(y, 20) + 38*max(0, y-20)
    # Using y_extra = max(0, y-20) and min(y,20) = y - y_extra:
    # => 50*x + 43*(y - y_extra) + 38*y_extra = 50*x + 43*y - 5*y_extra
    model.setObjective(50 * x + 43 * y - 5 * y_extra, GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(x + y <= total_shifts, "total_shifts")
    model.addConstr(5 * x + 6 * y <= energy_limit, "energy")
    model.addConstr(10 * x + 7 * y >= min_orders, "orders")
    model.addConstr(y >= min_scooter_shifts, "min_scooter_shifts")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Also return the optimal shifts for clarity
        bike_shifts = int(x.X)
        scooter_shifts = int(y.X)
        return model.objVal, bike_shifts, scooter_shifts
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_shifts()
    if result is not None:
        max_tips, bike_shifts, scooter_shifts = result
        print(f"Maximum Tips: ${max_tips}")
        print(f"Optimal bike shifts: {bike_shifts}")
        print(f"Optimal scooter shifts: {scooter_shifts}")
    else:
        print("No feasible solution found.")
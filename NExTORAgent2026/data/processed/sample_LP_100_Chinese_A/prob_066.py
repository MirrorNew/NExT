def optimize_shifts(total_shifts=40, energy_limit=230, min_orders=320, min_scooter_shifts=5):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Delivery_Shifts_Maximize_Tips_Nonlinear")

    # Decision variables: number of bike and scooter (motorbike) shifts
    x = model.addVar(vtype=GRB.INTEGER, name="bike_shifts", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="scooter_shifts", lb=0)

    # Set objective: maximize total tips
    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(50 * x + 43 * y, GRB.MAXIMIZE)
    model.setObjective(50 * x + 43 * y, GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(x + y <= total_shifts, "total_shifts")

    # Energy constraint:
    # Original linear form:
    # 5 * x + 6 * y <= energy_limit
    #
    # Now: bike energy is still linear (5 per bike shift),
    # but scooter (motorbike) energy has a nonlinear amplification:
    # if we schedule y scooter shifts, the scooter energy is (6y)^1.2
    # so the total energy constraint becomes:
    # 5 * x + (6 * y)^1.2 <= energy_limit

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(5 * x + 6 * y <= energy_limit, "energy")
    Y = model.addVar()
    model.addConstr(Y == 6 * y)
    YY = model.addVar()
    model.addGenConstrPow(Y,YY,1.2)
    model.addConstr(5 * x + YY <= energy_limit, "energy")

    # Orders constraint remains linear:
    model.addConstr(10 * x + 7 * y >= min_orders, "orders")

    # Minimum scooter (motorbike) shifts constraint remains linear:
    model.addConstr(y >= min_scooter_shifts, "min_scooter_shifts")

    # Optimize the (now nonlinear) model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None


# Example usage
if __name__ == "__main__":
    max_tips = optimize_shifts()
    if max_tips is not None:
        print(f"Maximum Tips: ${max_tips}")
    else:
        print("No feasible solution found.")
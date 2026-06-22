def optimize_car_oil_production(
    max_A=1345, max_B=346, max_C=1643,
    profit_max=10, profit_pro=15,
    content_A_max=(46, 13), content_B_max=(43, 4), content_C_max=(56, 45),
    overtime_fee=200, pro_threshold=20
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("OilProductionMaximizeProfit")

    # Decision variables: number of containers for each oil type
    x = model.addVar(vtype=GRB.INTEGER, name="OilMax")
    y = model.addVar(vtype=GRB.INTEGER, name="OilMaxPro")

    # Binary variable indicating whether overtime fee is triggered
    z = model.addVar(vtype=GRB.BINARY, name="OvertimeTrigger")

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(profit_max * x + profit_pro * y, GRB.MAXIMIZE)
    # Set objective: maximize profit, subtracting overtime fee if Pro exceeds threshold
    model.setObjective(profit_max * x + profit_pro * y - overtime_fee * z, GRB.MAXIMIZE)

    # Add resource constraints
    model.addConstr(content_A_max[0] * x + content_A_max[1] * y <= max_A, "ResourceA")
    model.addConstr(content_B_max[0] * x + content_B_max[1] * y <= max_B, "ResourceB")
    model.addConstr(content_C_max[0] * x + content_C_max[1] * y <= max_C, "ResourceC")

    # ❤ Non-linearity is introduced. ❤
    # (No such condition in the original model; we now link y with the binary z)
    # If y > pro_threshold, z must be 1 and the overtime fee applies.
    # If y <= pro_threshold, z can be 0 and the overtime fee need not apply.
    #
    # Use a big-M style linearization:
    #   y <= pro_threshold + M * z
    #   y >= pro_threshold + 1 - M * (1 - z)
    # Choose M as an upper bound on y; here we take a large enough constant.
    M = 1000

    # Ensure that if z = 0 then y <= pro_threshold
    model.addConstr(y <= pro_threshold + M * z, name="OvertimeUpperLink")
    # Ensure that if z = 1 then y >= pro_threshold + 1
    model.addConstr(y >= pro_threshold + 1 - M * (1 - z), name="OvertimeLowerLink")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        solution = {
            "OilMax": x.X,
            "OilMaxPro": y.X,
            "OvertimeTrigger": z.X,
            "MaxProfit": model.objVal,
        }
        return solution
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_car_oil_production()
    if result is not None:
        print(f"Optimal production plan:")
        print(f"  Oil Max      (x): {result['OilMax']}")
        print(f"  Oil Max Pro  (y): {result['OilMaxPro']}")
        print(f"  Overtime fee triggered (z): {int(result['OvertimeTrigger'])}")
        print(f"Maximum Profit: {result['MaxProfit']}")
    else:
        print("No feasible solution found.")
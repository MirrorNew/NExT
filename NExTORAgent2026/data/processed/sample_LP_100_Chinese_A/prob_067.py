def optimize_car_oil_production(
    max_A=1345, max_B=346, max_C=1643,
    profit_max=10, profit_pro=15,
    content_A_max=(46, 13), content_B_max=(43, 4), content_C_max=(56, 45)
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("OilProductionMaximizeProfit")

    # Decision variables: number of containers for each oil type
    x = model.addVar(vtype=GRB.INTEGER, name="OilMax")
    y = model.addVar(vtype=GRB.INTEGER, name="OilMaxPro")

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(profit_max * x + profit_pro * y, GRB.MAXIMIZE)
    # Introduce non-linear (actually mixed-integer nonlinear) objective:
    #   Maximize: 10 * x + 15 * (y ^ 1.2)
    Y = model.addVar()
    model.addGenConstrPow(y,Y,1.002)
    model.setObjective(
        profit_max * x + profit_pro * Y,
        GRB.MAXIMIZE
    )

    # Add resource constraints
    model.addConstr(content_A_max[0] * x + content_A_max[1] * y <= max_A, "ResourceA")
    model.addConstr(content_B_max[0] * x + content_B_max[1] * y <= max_B, "ResourceB")
    model.addConstr(content_C_max[0] * x + content_C_max[1] * y <= max_C, "ResourceC")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the maximum profit and corresponding production plan
        return {
            "max_profit": model.objVal,
            "OilMax": x.X,
            "OilMaxPro": y.X
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_car_oil_production()
    if result is not None:
        print(f"Maximum Profit: {result['max_profit']}")
        print(f"OilMax (barrels): {result['OilMax']}")
        print(f"OilMaxPro (barrels): {result['OilMaxPro']}")
    else:
        print("No feasible solution found.")
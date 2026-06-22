def optimize_meat_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Meat_Processing_Optimization")

    # Decision variables: number of batches of hams and pork ribs
    x = m.addVar(name="Hams", lb=0)
    y = m.addVar(name="Pork_Ribs", lb=0)

    # Auxiliary variable: number of pork rib batches above 500
    # This will capture max(y - 500, 0) in a linear way
    y_extra = m.addVar(name="Pork_Ribs_Above_500", lb=0)

    # Binary variable to activate the "above 500" regime
    z = m.addVar(vtype=GRB.BINARY, name="Above_500_Indicator")

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(150 * x + 300 * y, GRB.MAXIMIZE)

    # New (piecewise-linear) objective:
    # Profit = 150*x + 300*(y - y_extra) + 250*y_extra
    #       = 150*x + 300*y - 50*y_extra
    m.setObjective(150 * x + 300 * y - 50 * y_extra, GRB.MAXIMIZE)

    # Add constraints
    # Meat slicer constraint
    m.addConstr(4 * x + 2 * y <= 4000, name="Slicer_Time")
    # Meat packer constraint
    m.addConstr(2.5 * x + 3.5 * y <= 4000, name="Packer_Time")

    # Linearization of "y_extra = max(y - 500, 0)" using big-M
    M = 4000  # A safe upper bound on possible y (from time limits)

    # When z = 0, force y <= 500, hence y_extra = 0
    m.addConstr(y - 500 <= M * z, name="Above500_Activate")
    m.addConstr(y <= 500 + M * z, name="Cap_When_Not_Active")

    # Link y_extra to y and z:
    # y_extra >= y - 500  (only relevant when y > 500)
    m.addConstr(y_extra >= y - 500, name="Yextra_LB1")
    # y_extra <= y - 500 + M*(1 - z)
    m.addConstr(y_extra <= y - 500 + M * (1 - z), name="Yextra_UB1")
    # y_extra <= M*z  (forces y_extra = 0 when z = 0)
    m.addConstr(y_extra <= M * z, name="Yextra_UB2")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return detailed results
        return {
            "max_profit": m.objVal,
            "hams": x.X,
            "pork_ribs": y.X,
            "pork_ribs_above_500": y_extra.X,
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_meat_production()
    if result is not None:
        print(f"Maximum Profit: ${result['max_profit']}")
        print(f"Optimal Hams (batches): {result['hams']}")
        print(f"Optimal Pork Ribs (batches): {result['pork_ribs']}")
        print(f"Pork Ribs Above 500 (batches): {result['pork_ribs_above_500']}")
    else:
        print("No feasible solution found.")
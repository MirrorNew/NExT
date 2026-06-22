def optimize_crop_allocation(total_land=100,
                             profit_c=1500,
                             profit_w=1200,
                             profit_s=1800,
                             profit_so=1600,
                             synergy_coeff=10):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Crop_Profit_Maximization_With_Synergy")

    # Decision variables
    C = model.addVar(name="corn", lb=0)
    W = model.addVar(name="wheat", lb=0)
    S = model.addVar(name="soybeans", lb=0)
    M = model.addVar(name="sorghum", lb=0)

    Y = model.addVar(name="add", lb=0)
    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(
    #     profit_c * C + profit_w * W + profit_s * S + profit_so * M,
    #     GRB.MAXIMIZE)

    # Set objective: maximize profit with additional non-linear synergy term 10 * C * W * S
    model.addConstr(Y == C * W, "WheatSorghumRatio")
    model.setObjective(
        profit_c * C
        + profit_w * W
        + profit_s * S
        + profit_so * M
        + synergy_coeff * Y * S,
        GRB.MAXIMIZE
    )

    # Add constraints
    # Total land constraint
    model.addConstr(C + W + S + M <= total_land, "TotalLand")
    # Wheat equals three times sorghum
    model.addConstr(W == 3 * M, "WheatSorghumRatio")
    # Corn at least twice wheat
    model.addConstr(C >= 2 * W, "CornWheatRatio")
    # Soybeans at least half sorghum
    model.addConstr(S >= 0.5 * M, "SoySorghumRatio")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the total profit of the optimal solution and the decision variables
        return {
            "objective_value": model.objVal,
            "corn": C.X,
            "wheat": W.X,
            "soybeans": S.X,
            "sorghum": M.X
        }
    else:
        # No feasible solution found
        return None


if __name__ == "__main__":
    result = optimize_crop_allocation()
    if result is not None:
        print(f"Optimal total profit: {result['objective_value']}")
        print(f"Optimal allocation (acres):")
        print(f"  Corn (C):     {result['corn']}")
        print(f"  Wheat (W):    {result['wheat']}")
        print(f"  Soybeans (S): {result['soybeans']}")
        print(f"  Sorghum (M):  {result['sorghum']}")
    else:
        print("No feasible solution found.")
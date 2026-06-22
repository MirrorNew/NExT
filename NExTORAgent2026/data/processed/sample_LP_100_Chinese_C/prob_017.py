def optimize_toy_production(
        wood_available=890,
        steel_available=500,
        profit_truck=5,
        profit_airplane=10,
        profit_boat=8,
        profit_train=7,
        M=1000  # Big-M value
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create model
    model = gp.Model("HausToysOptimization")
    model.setParam('OutputFlag', 0)  # Silence output

    # Decision variables
    T = model.addVar(vtype=GRB.INTEGER, name="T")  # Trucks
    A = model.addVar(vtype=GRB.INTEGER, name="A")  # Airplanes
    B = model.addVar(vtype=GRB.INTEGER, name="B")  # Boats
    R = model.addVar(vtype=GRB.INTEGER, name="R")  # Trains

    # Binary variables for logical conditions
    y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
    y_R = model.addVar(vtype=GRB.BINARY, name="y_R")
    y_B = model.addVar(vtype=GRB.BINARY, name="y_B")

    # ❤ Non-linearity is introduced. ❤
    # Binary variable indicating whether airplane production exceeds 20
    z_A = model.addVar(vtype=GRB.BINARY, name="z_A")

    # Variable for number of airplanes above 20 (only positive if A > 20)
    A_extra = model.addVar(vtype=GRB.INTEGER, name="A_extra", lb=0)

    # Fixed maintenance cost (incurred once if A > 20)
    fixed_maintenance_cost = 100

    # Objective: maximize profit (base profit minus overtime and fixed cost)
    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(
    #     profit_truck * T + profit_airplane * A + profit_boat * B +
    #     profit_train * R, GRB.MAXIMIZE)
    model.setObjective(
        profit_truck * T
        + profit_airplane * A
        + profit_boat * B
        + profit_train * R
        - 2 * A_extra                    # overtime labor: $2 per airplane above 20
        - fixed_maintenance_cost * z_A,  # fixed $100 if A > 20
        GRB.MAXIMIZE
    )

    # Resource constraints
    model.addConstr(12 * T + 20 * A + 15 * B + 10 * R <= wood_available,
                    "Wood")
    model.addConstr(6 * T + 3 * A + 5 * B + 4 * R <= steel_available, "Steel")

    # Boats cannot exceed trains
    model.addConstr(B <= R, "BoatTrainLimit")

    # Logical constraints for trucks and trains (mutual exclusivity)
    model.addConstr(T <= M * y_T, "TruckLogical")
    model.addConstr(R <= M * y_R, "TrainLogical")
    model.addConstr(y_T + y_R <= 1, "MutualExclusion")

    # Logical constraints for boats and airplanes
    model.addConstr(A >= y_B, "AirplaneBoat")
    model.addConstr(B <= y_B * M, "BoatBinary")

    # ------------------------------
    # Linearization of the nonlinear cost effect for airplanes
    # ------------------------------
    # Link A_extra and z_A to capture "A > 20" region:
    #
    # If z_A = 0 (A <= 20), then:
    #   A_extra must be 0.
    # If z_A = 1 (A > 20), then:
    #   A_extra = A - 20 (within big-M bounds).

    # A_extra cannot exceed total airplanes
    model.addConstr(A_extra <= A, name="Aextra_le_A")

    # If z_A = 0 → A_extra <= 0; if z_A = 1 → A_extra can go up to M
    model.addConstr(A_extra <= M * z_A, name="Aextra_bigM_upper")

    # Lower bound: A_extra ≥ A - 20 when z_A = 1
    model.addConstr(A_extra >= A - 20 - M * (1 - z_A),
                    name="Aextra_lb_link")

    # When z_A = 0 → ensure A ≤ 20
    model.addConstr(A <= 20 + M * z_A, name="A_le_20_or_bigM")

    # Also, A_extra cannot be negative
    model.addConstr(A_extra >= 0, name="Aextra_nonneg")

    # Optimize
    model.optimize()

    # Check feasibility
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None


if __name__ == "__main__":
    result = optimize_toy_production()
    print(f"Optimal profit: {result}")
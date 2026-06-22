import gurobipy as gp

# Solves the Orissa Electric Holding Company unit commitment problem
# and prints the minimum total cost and schedule.
if __name__ == "__main__":
    # -----------------------------
    # 1. Parameters (from Parameters List)
    # -----------------------------
    company_name = "Orissa Electric Holding Company"
    company_location = "eastern Mexico"

    num_power_plants = 17
    plant_names = [
        'Biomass 1', 'Biomass 2',
        'Combined Cycle 1', 'Combined Cycle 2', 'Combined Cycle 3', 'Combined Cycle 4',
        'Wind Energy 1', 'Wind Energy 2', 'Wind Energy 3', 'Wind Energy 4',
        'Hydropower 1', 'Hydropower 2', 'Hydropower3',
        'Solar1', 'Solar2',
        'Coal1', 'Coal2'
    ]

    num_periods = 4
    periods = [1, 2, 3, 4]

    demands_MWh = [11832.52, 14467.92, 16661.1, 15434.44]
    demands = {t: demands_MWh[t - 1] for t in periods}

    min_power_MWh = [78.75, 100.0, 742.5, 800.0, 700.0, 750.0,
                     246.0, 200.0, 220.0, 250.0,
                     50.0, 60.0, 55.0,
                     20.0, 20.0,
                     500.0, 550.0]

    max_power_MWh = [295.14, 300.0, 2883.9, 3000.0, 2800.0, 2900.0,
                     294.9, 350.0, 330.0, 340.0,
                     500.0, 600.0, 550.0,
                     400.0, 350.0,
                     2500.0, 2600.0]

    change_cost_USD_per_MWh = [3.94, 4.1, 2.72, 2.85, 2.65, 2.8,
                               0.0, 0.0, 0.0, 0.0,
                               1.5, 1.8, 1.6,
                               0.0, 0.0,
                               5.0, 4.8]

    fixed_cost_USD = [265.524, 270.0, 92.28, 95.0, 90.0, 93.0,
                      149.778, 160.0, 155.0, 162.0,
                      60.0, 65.0, 62.0,
                      120.0, 110.0,
                      320.0, 315.0]

    startup_cost_USD = [0.0, 0.0, 500.0, 500.0, 500.0, 500.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0,
                        500.0, 480.0]

    shutdown_cost_USD = [330.0, 340.0, 210.0, 215.0, 205.0, 208.0,
                         240.0, 250.0, 245.0, 255.0,
                         100.0, 120.0, 110.0,
                         150.0, 140.0,
                         400.0, 390.0]

    # Map plant parameters into dictionaries keyed by plant name
    min_power = {i: v for i, v in zip(plant_names, min_power_MWh)}
    max_power = {i: v for i, v in zip(plant_names, max_power_MWh)}
    change_cost = {i: v for i, v in zip(plant_names, change_cost_USD_per_MWh)}
    fixed_cost = {i: v for i, v in zip(plant_names, fixed_cost_USD)}
    startup_cost = {i: v for i, v in zip(plant_names, startup_cost_USD)}
    shutdown_cost = {i: v for i, v in zip(plant_names, shutdown_cost_USD)}

    # Combined cycle plants set
    combined_cycle_plants = [
        'Combined Cycle 1', 'Combined Cycle 2',
        'Combined Cycle 3', 'Combined Cycle 4'
    ]

    # Minimum number of running units
    min_running_units = 8

    # Initial status and power (all units off and zero output)
    initial_status = {i: 0 for i in plant_names}
    initial_power = {i: 0.0 for i in plant_names}

    # -----------------------------
    # 2. Create model
    # -----------------------------
    model = gp.Model("Orissa_Unit_Commitment")

    # -----------------------------
    # 3. Decision variables
    # -----------------------------
    # u[i,t]: on/off status
    u = model.addVars(plant_names, periods, vtype=gp.GRB.BINARY, name="u")
    # P[i,t]: power output
    P = model.addVars(plant_names, periods, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="P")
    # Startup and shutdown indicators
    y = model.addVars(plant_names, periods, vtype=gp.GRB.BINARY, name="y")
    z = model.addVars(plant_names, periods, vtype=gp.GRB.BINARY, name="z")
    # Power change positive and negative parts
    dP_pos = model.addVars(plant_names, periods, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="dP_pos")
    dP_neg = model.addVars(plant_names, periods, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="dP_neg")

    # -----------------------------
    # 4. Constraints
    # -----------------------------

    # Generation bounds
    for i in plant_names:
        for t in periods:
            model.addConstr(P[i, t] <= max_power[i] * u[i, t],
                            name=f"GenMax_{i}_{t}")
            model.addConstr(P[i, t] >= min_power[i] * u[i, t],
                            name=f"GenMin_{i}_{t}")

    # Startup / shutdown logical relation
    for i in plant_names:
        # t = 1 uses initial status
        model.addConstr(u[i, 1] - initial_status[i] == y[i, 1] - z[i, 1],
                        name=f"Logic_{i}_1")
        # t = 2,3,4
        for t in [2, 3, 4]:
            model.addConstr(u[i, t] - u[i, t - 1] == y[i, t] - z[i, t],
                            name=f"Logic_{i}_{t}")

    # Power change balance
    for i in plant_names:
        # t = 1 from initial power
        model.addConstr(P[i, 1] - initial_power[i] == dP_pos[i, 1] - dP_neg[i, 1],
                        name=f"dPBal_{i}_1")
        # t = 2,3,4 from previous period
        for t in [2, 3, 4]:
            model.addConstr(P[i, t] - P[i, t - 1] == dP_pos[i, t] - dP_neg[i, t],
                            name=f"dPBal_{i}_{t}")

    # Demand satisfaction
    for t in periods:
        model.addConstr(
            gp.quicksum(P[i, t] for i in plant_names) >= demands[t],
            name=f"Demand_{t}"
        )

    # Combined cycle must keep running if on: u[i,t] <= u[i,t+1]
    for i in combined_cycle_plants:
        for t in [1, 2, 3]:
            model.addConstr(u[i, t] <= u[i, t + 1],
                            name=f"CCRun_{i}_{t}")

    # Minimum number of running units
    for t in periods:
        model.addConstr(
            gp.quicksum(u[i, t] for i in plant_names) >= min_running_units,
            name=f"MinUnits_{t}"
        )

    # -----------------------------
    # 5. Objective function
    # -----------------------------
    obj = gp.LinExpr()
    for i in plant_names:
        for t in periods:
            obj += fixed_cost[i] * u[i, t]
            obj += startup_cost[i] * y[i, t]
            obj += shutdown_cost[i] * z[i, t]
            obj += change_cost[i] * (dP_pos[i, t] + dP_neg[i, t])

    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    # -----------------------------
    # 6. Solve
    # -----------------------------
    model.optimize()

    # -----------------------------
    # 7. Print results
    # -----------------------------
    if model.Status == gp.GRB.OPTIMAL:
        print(f"Company: {company_name} ({company_location})")
        print(f"Optimal objective (minimum total cost): {model.ObjVal:.4f} USD")

        for t in periods:
            print(f"\nPeriod {t}: Demand = {demands[t]} MWh")
            total_gen = 0.0
            num_on = 0
            for i in plant_names:
                if u[i, t].X > 0.5:
                    num_on += 1
                    p_val = P[i, t].X
                    total_gen += p_val
                    y_val = y[i, t].X
                    z_val = z[i, t].X
                    print(
                        f"  {i:18s} ON,  P = {p_val:8.2f} MWh, "
                        f"Startup = {int(round(y_val))}, Shutdown = {int(round(z_val))}"
                    )
            print(f"  Total generation: {total_gen:.2f} MWh, units on: {num_on}")

        # FinalAnswer must be the minimum cost (objective value)
        FinalAnswer = model.ObjVal
        print(f"FinalAnswer=【{FinalAnswer}】")
    else:
        print("No optimal solution found. Gurobi status code:", model.Status)
        # If no optimal solution, still output something for FinalAnswer
        FinalAnswer = float('nan')
        print(f"FinalAnswer=【{FinalAnswer}】")
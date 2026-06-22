import gurobipy as gp

# Complete and reliable Gurobi code for the described MILP

# Question: Minimize costs and give the minimum total cost over the 15 periods.


# -----------------------------
# Data and parameters (from Parameters List)
# -----------------------------
planning_horizon_periods = 15
safety_stock_ratio = 0.3

entities = [
    'Central Warehouse',
    'Regional Warehouse 1',
    'Regional Warehouse 2',
    'Retail Store 1',
    'Retail Store 2',
    'Retail Store 3',
    'Retail Store 4'
]
stores = ['Retail Store 1', 'Retail Store 2', 'Retail Store 3', 'Retail Store 4']
regional_warehouses = ['Regional Warehouse 1', 'Regional Warehouse 2']

initial_stock = {
    'Central Warehouse': 1200,
    'Regional Warehouse 1': 500,
    'Regional Warehouse 2': 400,
    'Retail Store 1': 350,
    'Retail Store 2': 450,
    'Retail Store 3': 500,
    'Retail Store 4': 600
}

demand_per_period = {
    'Central Warehouse': None,
    'Regional Warehouse 1': None,
    'Regional Warehouse 2': None,
    'Retail Store 1': 50,
    'Retail Store 2': 60,
    'Retail Store 3': 70,
    'Retail Store 4': 80
}

# Costs
ordering_cost_per_order = 30.0
warehouse_holding_cost_per_unit_period = 0.2
retail_holding_cost_per_unit_period = 0.6
min_orders_central_to_R1 = 10

# Transport Route Costs and Capacities (Table_2_Transport_Route_Costs)
cap_C_r = {
    'Regional Warehouse 1': 1000,
    'Regional Warehouse 2': 1000
}
ctrans_C_r = {
    'Regional Warehouse 1': 0.55,
    'Regional Warehouse 2': 0.22
}

cap_r_s = {}
ctrans_r_s = {}

# Regional warehouse 1
cap_r_s[('Regional Warehouse 1', 'Retail Store 1')] = 500
cap_r_s[('Regional Warehouse 1', 'Retail Store 2')] = 500
cap_r_s[('Regional Warehouse 1', 'Retail Store 3')] = 500
cap_r_s[('Regional Warehouse 1', 'Retail Store 4')] = 500

ctrans_r_s[('Regional Warehouse 1', 'Retail Store 1')] = 0.22
ctrans_r_s[('Regional Warehouse 1', 'Retail Store 2')] = 0.20
ctrans_r_s[('Regional Warehouse 1', 'Retail Store 3')] = 0.32
ctrans_r_s[('Regional Warehouse 1', 'Retail Store 4')] = 0.38

# Regional warehouse 2
cap_r_s[('Regional Warehouse 2', 'Retail Store 1')] = 500
cap_r_s[('Regional Warehouse 2', 'Retail Store 2')] = 500
cap_r_s[('Regional Warehouse 2', 'Retail Store 3')] = 500
cap_r_s[('Regional Warehouse 2', 'Retail Store 4')] = 500

ctrans_r_s[('Regional Warehouse 2', 'Retail Store 1')] = 0.68
ctrans_r_s[('Regional Warehouse 2', 'Retail Store 2')] = 0.52
ctrans_r_s[('Regional Warehouse 2', 'Retail Store 3')] = 0.34
ctrans_r_s[('Regional Warehouse 2', 'Retail Store 4')] = 0.10

# Safety stock per period at each store (0.3 * demand)
safety_stock_per_period_store = {
    s: safety_stock_ratio * demand_per_period[s] for s in stores
}

# Provided totals
total_demand_over_horizon_all_stores = 3900

# Big-M for order quantity
M_C_r = {
    'Regional Warehouse 1': total_demand_over_horizon_all_stores,
    'Regional Warehouse 2': total_demand_over_horizon_all_stores
}
# Big-M for penalty logic
M_pen = planning_horizon_periods  # 15

# Penalty cost coefficient: not specified in Parameters List; set to 0 to reflect "fee only if violated",
# but since value is unknown we keep it as 0, so model will avoid using z only if it doesn't change cost.
c_pen = 0.0


# -----------------------------
# Build and solve model
# -----------------------------
def main():
    periods = range(1, planning_horizon_periods + 1)

    # Create model
    model = gp.Model("Four_Level_Supply_Chain")

    # -----------------------------
    # Decision variables
    # -----------------------------

    # Inventory levels at end of period t
    I_C = model.addVars(periods, name="I_C", lb=0.0)  # central
    I_R = model.addVars(regional_warehouses, periods, name="I_R", lb=0.0)
    I_S = model.addVars(stores, periods, name="I_S", lb=0.0)

    # Shipments
    Q_C_R = model.addVars(regional_warehouses, periods, name="Q_C_R", lb=0.0)
    Q_R_S = model.addVars(regional_warehouses, stores, periods, name="Q_R_S", lb=0.0)

    # Order quantities from central to RW (logical replenishment amounts)
    Q_ord = model.addVars(regional_warehouses, periods, name="Q_ord", lb=0.0)

    # Binary: 1 if region r orders in period t
    y = model.addVars(regional_warehouses, periods, vtype=gp.GRB.BINARY, name="y")

    # Penalty indicator and cost
    z = model.addVar(vtype=gp.GRB.BINARY, name="z")
    C_pen = model.addVar(lb=0.0, name="C_pen")

    # -----------------------------
    # Constraints
    # -----------------------------

    # 1) Central warehouse inventory balance
    for t in periods:
        if t == 1:
            prev_I_C = initial_stock['Central Warehouse']
        else:
            prev_I_C = I_C[t - 1]

        model.addConstr(
            I_C[t] == prev_I_C - gp.quicksum(Q_C_R[r, t] for r in regional_warehouses),
            name=f"InvBal_C_{t}"
        )

    # 2) Regional warehouse inventory balance
    # Use inflow from central: Q_C_R[r,t]; Q_ord is logical ordering amount (for cost and linking),
    # but inflow to inventory is the actual shipment Q_C_R[r,t].
    for r in regional_warehouses:
        for t in periods:
            if t == 1:
                prev_I_R = initial_stock[r]
            else:
                prev_I_R = I_R[r, t - 1]

            model.addConstr(
                I_R[r, t] == prev_I_R + Q_C_R[r, t] - gp.quicksum(Q_R_S[r, s, t] for s in stores),
                name=f"InvBal_R_{r}_{t}"
            )

    # 3) Store inventory balance and safety stock
    for s in stores:
        D_s = demand_per_period[s]
        safety_s = safety_stock_per_period_store[s]

        for t in periods:
            if t == 1:
                prev_I_S = initial_stock[s]
            else:
                prev_I_S = I_S[s, t - 1]

            model.addConstr(
                I_S[s, t] == prev_I_S +
                gp.quicksum(Q_R_S[r, s, t] for r in regional_warehouses) -
                D_s,
                name=f"InvBal_S_{s}_{t}"
            )

            # Safety stock: I^S_{s,t} >= 0.3 * D_s
            model.addConstr(
                I_S[s, t] >= safety_s,
                name=f"Safety_{s}_{t}"
            )

    # 4) Transport capacity constraints: central -> regional
    for r in regional_warehouses:
        for t in periods:
            model.addConstr(
                Q_C_R[r, t] <= cap_C_r[r],
                name=f"Cap_C_R_{r}_{t}"
            )

    # 5) Transport capacity constraints: regional -> store
    for r in regional_warehouses:
        for s in stores:
            for t in periods:
                model.addConstr(
                    Q_R_S[r, s, t] <= cap_r_s[(r, s)],
                    name=f"Cap_R_S_{r}_{s}_{t}"
                )

    # 6) Order-quantity–binary linking via indicator constraints (no big-M)
    # If y[r,t] == 0 -> Q_ord[r,t] == 0
    # If y[r,t] == 1 -> Q_ord[r,t] <= M_C_r[r]
    for r in regional_warehouses:
        for t in periods:
            # y[r,t] = 0 => Q_ord[r,t] = 0
            model.addGenConstrIndicator(
                y[r, t], 0, Q_ord[r, t] == 0.0,
                name=f"Ind_NoOrder_QordZero_{r}_{t}"
            )
            # y[r,t] = 1 => Q_ord[r,t] <= M_C_r[r]
            model.addGenConstrIndicator(
                y[r, t], 1, Q_ord[r, t] <= M_C_r[r],
                name=f"Ind_Order_QordUB_{r}_{t}"
            )

    # 7) Link actual shipment to order quantity:
    # Q_C_R[r,t] <= Q_ord[r,t]   (cannot ship more than ordered)
    for r in regional_warehouses:
        for t in periods:
            model.addConstr(
                Q_C_R[r, t] <= Q_ord[r, t],
                name=f"Ship_le_Qord_{r}_{t}"
            )

    # 8) Order count requirement for RW1 with possible penalty
    r1 = 'Regional Warehouse 1'
    # Indicator form: if z == 0 then sum_t y[r1,t] >= min_orders_central_to_R1
    model.addGenConstrIndicator(
        z, 0,
        gp.quicksum(y[r1, t] for t in periods) >= min_orders_central_to_R1,
        name="Ind_NoPenalty_MinOrdersRW1"
    )
    # z is binary; if z=1, this indicator is inactive and constraint can be violated with penalty

    # 9) Penalty cost linking: if z == 1 then C_pen == c_pen; if z == 0 then C_pen == 0
    # z = 0 => C_pen == 0
    model.addGenConstrIndicator(
        z, 0, C_pen == 0.0,
        name="Ind_z0_Cpen0"
    )
    # z = 1 => C_pen == c_pen
    model.addGenConstrIndicator(
        z, 1, C_pen == c_pen,
        name="Ind_z1_Cpen_cpen"
    )

    # -----------------------------
    # Objective function
    # -----------------------------
    # Min Z = ordering + holding + transport + penalty

    # Ordering costs: pay ordering cost when a region places an order
    ordering_cost_expr = gp.quicksum(
        ordering_cost_per_order * y[r, t]
        for r in regional_warehouses for t in periods
    )

    # Holding costs
    holding_cost_expr = gp.quicksum(
        warehouse_holding_cost_per_unit_period * I_C[t] for t in periods
    )
    holding_cost_expr += gp.quicksum(
        warehouse_holding_cost_per_unit_period * I_R[r, t]
        for r in regional_warehouses for t in periods
    )
    holding_cost_expr += gp.quicksum(
        retail_holding_cost_per_unit_period * I_S[s, t]
        for s in stores for t in periods
    )

    # Transportation costs: central -> regional
    transport_cost_expr = gp.quicksum(
        ctrans_C_r[r] * Q_C_R[r, t]
        for r in regional_warehouses for t in periods
    )
    # Transportation costs: regional -> store
    transport_cost_expr += gp.quicksum(
        ctrans_r_s[(r, s)] * Q_R_S[r, s, t]
        for r in regional_warehouses for s in stores for t in periods
    )

    penalty_cost_expr = C_pen

    model.setObjective(
        ordering_cost_expr + holding_cost_expr + transport_cost_expr + penalty_cost_expr,
        gp.GRB.MINIMIZE
    )

    # -----------------------------
    # Solve
    # -----------------------------
    model.optimize()

    # -----------------------------
    # Print results
    # -----------------------------
    if model.status == gp.GRB.OPTIMAL:
        total_cost = model.objVal
        print(f"Optimal objective value (minimum total cost): {total_cost:.4f}")

        print("\nOrder decisions y[r,t]:")
        for r in regional_warehouses:
            decisions = [int(round(y[r, t].X)) for t in periods]
            print(f"{r}: {decisions}")

        print("\nPenalty indicator z:", int(round(z.X)))
        print("Penalty cost C_pen:", C_pen.X)

        # Final answer required by instruction:
        print(f"FinalAnswer=【{total_cost:.4f}】")
    else:
        print("No optimal solution found. Status code:", model.status)
        # If no optimal value, still print FinalAnswer as NaN
        print("FinalAnswer=【NaN】")


if __name__ == "__main__":
    main()
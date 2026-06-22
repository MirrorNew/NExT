import gurobipy as gp
from gurobipy import GRB

def solve_supply_chain():
    # 1. Initialize Model
    model = gp.Model("SupplyChainOptimization")

    # 2. Parameters and Sets
    I = [1, 2]  # Plants A1, A2
    J_all = [1, 2, 3, 4, 5, 6]  # Warehouses B1-B6
    K = [1, 2, 3, 4, 5, 6]  # Users C1-C6

    # Supply and Demand
    S = {1: 150000, 2: 200000}
    D = {1: 50000, 2: 10000, 3: 40000, 4: 35000, 5: 60000, 6: 20000}

    # Base Capacities (T)
    T_base = {1: 70000, 2: 50000, 3: 100000, 4: 40000, 5: 30000, 6: 25000}

    # Financial Parameters
    Inv_B5 = 1200000
    Inv_B6 = 400000
    Inv_ExpB2 = 300000
    Cap_ExpB2_Add = 20000
    Save_B1 = 100000
    Save_B4 = 50000
    
    # Pre-assignments List from parameters
    # Indices: 0(dummy), 1(C1), 2(C2), 3(C3), 4(C4), 5(C5), 6(C6)
    pre_assign_list = [[], [['A', 1]], [['B', 1]], [], [], [['B', 2]], [['B', 3], ['B', 4]]]

    # 3. Cost Data Parsing
    # Initialize costs to infinity (or a sufficiently large number) to represent forbidden arcs
    BIG_M_COST = 1e9
    c_w = {(i, j): BIG_M_COST for i in I for j in J_all}
    c_z = {(i, k): BIG_M_COST for i in I for k in K}
    c_y = {(j, k): BIG_M_COST for j in J_all for k in K}

    # Table 1 Data (A1, A2, B1, B2, B3)
    # Note: Row 3 (Index 3) corresponds to B2 based on context analysis of length and values
    table1 = [
        ['A1', [50, 50, 100, 20, 100, None, 150, 200, None, 100]],
        ['A2', [None, 30, 50, 20, 200, None, None, None, None, None]],
        ['B1', [None, None, None, None, None, 150, 50, 150, None, 100]],
        [None, [None, None, 100, 50, 50, 100, 50, None, None]],
        ['B3', [None, None, None, None, None, 150, 200, None, 50, 150]]
    ]

    # Parse Row 0: A1
    vals_a1 = table1[0][1]
    # Indices 0-3 -> B1-B4
    for idx, cost in enumerate(vals_a1[0:4]):
        if cost is not None: c_w[(1, idx + 1)] = cost
    # Indices 4-9 -> C1-C6
    for idx, cost in enumerate(vals_a1[4:10]):
        if cost is not None: c_z[(1, idx + 1)] = cost

    # Parse Row 1: A2
    vals_a2 = table1[1][1]
    for idx, cost in enumerate(vals_a2[0:4]):
        if cost is not None: c_w[(2, idx + 1)] = cost
    for idx, cost in enumerate(vals_a2[4:10]):
        if cost is not None: c_z[(2, idx + 1)] = cost

    # Parse Row 2: B1 -> Users
    vals_b1 = table1[2][1]
    for idx, cost in enumerate(vals_b1[4:10]):
        if cost is not None: c_y[(1, idx + 1)] = cost

    # Parse Row 3: B2 -> Users (Special handling for length 9 list)
    # Mapped indices 2,3,4,5,6 correspond to C1, C2, C3, C4, C5
    vals_b2 = table1[3][1]
    b2_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5} # list_index : user_k
    for list_idx, user_k in b2_map.items():
        if list_idx < len(vals_b2) and vals_b2[list_idx] is not None:
            c_y[(2, user_k)] = vals_b2[list_idx]

    # Parse Row 4: B3 -> Users
    vals_b3 = table1[4][1]
    for idx, cost in enumerate(vals_b3[4:10]):
        if cost is not None: c_y[(3, idx + 1)] = cost

    # Table 2 Data (A1/A2 to New Warehouses, New Warehouses to Users)
    table2 = [
        ['A1', [60, 40, None, None, None, None, None, None]],
        ['A2', [40, 30, None, None, None, None, None, None]],
        ['B5', [None, None, 120, 60, 40, None, 30, 80]],
        ['B6', [None, None, None, 40, None, 50, 60, 90]]
    ]

    # A1 -> B5, B6
    vals_a1_new = table2[0][1]
    if vals_a1_new[0] is not None: c_w[(1, 5)] = vals_a1_new[0]
    if vals_a1_new[1] is not None: c_w[(1, 6)] = vals_a1_new[1]

    # A2 -> B5, B6
    vals_a2_new = table2[1][1]
    if vals_a2_new[0] is not None: c_w[(2, 5)] = vals_a2_new[0]
    if vals_a2_new[1] is not None: c_w[(2, 6)] = vals_a2_new[1]

    # B5 -> Users (Indices 2-7 map to C1-C6)
    vals_b5 = table2[2][1]
    for idx, cost in enumerate(vals_b5[2:8]):
        if cost is not None: c_y[(5, idx + 1)] = cost

    # B6 -> Users
    vals_b6 = table2[3][1]
    for idx, cost in enumerate(vals_b6[2:8]):
        if cost is not None: c_y[(6, idx + 1)] = cost

    # 4. Variables
    w = model.addVars(I, J_all, vtype=GRB.CONTINUOUS, name="w")
    z = model.addVars(I, K, vtype=GRB.CONTINUOUS, name="z")
    y = model.addVars(J_all, K, vtype=GRB.CONTINUOUS, name="y")
    
    # Binary design variables
    u = model.addVars(J_all, vtype=GRB.BINARY, name="u") # 1 if warehouse j is open
    e2 = model.addVar(vtype=GRB.BINARY, name="e2")       # 1 if B2 expanded

    # 5. Objective Function
    # Total Cost = Transportation + Investments - Savings
    # Savings are modeled as negative costs or opportunity costs.
    # Cost = Trans + Inv_B5*u5 + Inv_B6*u6 + Inv_ExpB2*e2 
    #        + Save_B1*(u1) + Save_B4*(u4) - (Save_B1 + Save_B4)
    # Explanation: If u1=1 (open), cost is +Save_B1 relative to closed state. Subtract constant to align with "saving" definition.
    
    obj = gp.LinExpr()
    
    # Transportation
    for i in I:
        for j in J_all:
            if c_w[(i,j)] < BIG_M_COST:
                obj += c_w[(i,j)] * w[i,j]
    for i in I:
        for k in K:
            if c_z[(i,k)] < BIG_M_COST:
                obj += c_z[(i,k)] * z[i,k]
    for j in J_all:
        for k in K:
            if c_y[(j,k)] < BIG_M_COST:
                obj += c_y[(j,k)] * y[j,k]

    # Fixed Costs & Savings
    obj += Inv_B5 * u[5] + Inv_B6 * u[6] + Inv_ExpB2 * e2
    obj += Save_B1 * u[1] + Save_B4 * u[4]
    obj -= (Save_B1 + Save_B4)

    model.setObjective(obj, GRB.MINIMIZE)

    # 6. Constraints

    # Plant Capacity
    for i in I:
        model.addConstr(gp.quicksum(w[i,j] for j in J_all) + gp.quicksum(z[i,k] for k in K) <= S[i], f"PlantCap_{i}")

    # Flow Conservation at Warehouses
    for j in J_all:
        model.addConstr(gp.quicksum(w[i,j] for i in I) == gp.quicksum(y[j,k] for k in K), f"FlowBal_{j}")

    # Demand Satisfaction
    for k in K:
        model.addConstr(gp.quicksum(z[i,k] for i in I) + gp.quicksum(y[j,k] for j in J_all) >= D[k], f"Demand_{k}")

    # Warehouse Throughput Capacity
    # T1=70000u1, T2=50000u2+20000e2, etc.
    model.addConstr(gp.quicksum(y[1,k] for k in K) <= T_base[1] * u[1], "Cap_B1")
    model.addConstr(gp.quicksum(y[2,k] for k in K) <= T_base[2] * u[2] + Cap_ExpB2_Add * e2, "Cap_B2")
    model.addConstr(gp.quicksum(y[3,k] for k in K) <= T_base[3] * u[3], "Cap_B3")
    model.addConstr(gp.quicksum(y[4,k] for k in K) <= T_base[4] * u[4], "Cap_B4")
    model.addConstr(gp.quicksum(y[5,k] for k in K) <= T_base[5] * u[5], "Cap_B5")
    model.addConstr(gp.quicksum(y[6,k] for k in K) <= T_base[6] * u[6], "Cap_B6")

    # Operational Logic
    model.addConstr(u[2] == 1, "B2_AlwaysOpen")
    model.addConstr(u[3] == 1, "B3_AlwaysOpen")
    model.addConstr(e2 <= u[2], "ExpandOnlyIfOpen") # Redundant but good practice
    model.addConstr(gp.quicksum(u[j] for j in J_all) <= 4, "Max4Warehouses")

    # Route Availability (Forbidden Arcs)
    for i in I:
        for j in J_all:
            if c_w[(i,j)] >= BIG_M_COST: model.addConstr(w[i,j] == 0)
    for i in I:
        for k in K:
            if c_z[(i,k)] >= BIG_M_COST: model.addConstr(z[i,k] == 0)
    for j in J_all:
        for k in K:
            if c_y[(j,k)] >= BIG_M_COST: model.addConstr(y[j,k] == 0)

    # User Preferences (PreAssignment)
    for k_idx in range(1, 7): # Users 1..6
        if k_idx >= len(pre_assign_list): continue
        reqs = pre_assign_list[k_idx]
        if not reqs: continue
        
        # Determine allowed sources
        allowed_plants = set()
        allowed_warehouses = set()
        for rtype, ridx in reqs:
            if rtype == 'A': allowed_plants.add(ridx)
            if rtype == 'B': allowed_warehouses.add(ridx)
            
        # Enforce restrictions
        for i in I:
            if i not in allowed_plants:
                model.addConstr(z[i, k_idx] == 0, f"Restr_z_{i}_{k_idx}")
        for j in J_all:
            if j not in allowed_warehouses:
                model.addConstr(y[j, k_idx] == 0, f"Restr_y_{j}_{k_idx}")

    # 7. Solve and Output
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"FinalAnswer=【{model.objVal}】")
    else:
        print("FinalAnswer=【No Solution】")

if __name__ == "__main__":
    solve_supply_chain()
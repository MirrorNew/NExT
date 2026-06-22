import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Already imported above)

def solve_traffic_problem():
    # 2. Define all parameter matrices and data inputs.
    params = {
        'flow_threshold_A': 900,
        'capacity_threshold': 0.9,
        'total_vehicles': 2100,
        'idle_time': {'A': 10, 'B': 12},
        'travel_time_coefs': {'A': {'base': 10, 'coef': 0.0001}, 'B': {'base': 12, 'coef': 8e-05}},
        'additional_wait_rate': 0.05,
        'capacity': {'A': 1000, 'B': 1200}
    }

    # Calculate derived parameters
    threshold_A = params['flow_threshold_A']
    threshold_B = params['capacity']['B'] * params['capacity_threshold']  # 1200 * 0.9 = 1080
    penalty_factor = 1.0 + params['additional_wait_rate']  # 1.05

    # Initialize Model
    model = gp.Model("Traffic_Optimization")
    model.Params.NonConvex = 2  # Required for quadratic constraints and objective

    # 3. Create decision variables.
    f_A = model.addVar(lb=0, ub=params['capacity']['A'], vtype=GRB.CONTINUOUS, name="f_A")
    f_B = model.addVar(lb=0, ub=params['capacity']['B'], vtype=GRB.CONTINUOUS, name="f_B")
    
    # 4. Create auxiliary substitution variables.
    # Variables for f^2
    fA_sq = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="fA_sq")
    fB_sq = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="fB_sq")
    
    # Variables for Travel Time
    T_A = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_A")
    T_B = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_B")
    
    # Indicator variables for congestion (1 if congested/over threshold, 0 otherwise)
    y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
    y_B = model.addVar(vtype=GRB.BINARY, name="y_B")

    # 5. Set up the constraints.
    
    # Flow conservation
    model.addConstr(f_A + f_B == params['total_vehicles'], "Flow_Conservation")
    
    # Auxiliary constraints for squares: fA_sq = f_A^2
    model.addGenConstrPow(f_A, fA_sq, 2.0, "Pow_A")
    model.addGenConstrPow(f_B, fB_sq, 2.0, "Pow_B")
    
    # --- Road A Logic ---
    # Congestion Definition: y_A = 1 <-> f_A >= 900
    model.addGenConstrIndicator(y_A, 1, f_A >= threshold_A, name="Congestion_Def_A_1")
    model.addGenConstrIndicator(y_A, 0, f_A <= threshold_A, name="Congestion_Def_A_0")
    
    # Travel Time Calculation
    # Base expression: 10 + 0.0001 * fA^2
    base_time_A_expr = params['travel_time_coefs']['A']['base'] + params['travel_time_coefs']['A']['coef'] * fA_sq
    
    # If y_A=1 (Congested): T_A = 1.05 * Base
    model.addGenConstrIndicator(y_A, 1, T_A == penalty_factor * base_time_A_expr, name="Time_A_Congested")
    # If y_A=0 (Normal): T_A = Base
    model.addGenConstrIndicator(y_A, 0, T_A == base_time_A_expr, name="Time_A_Normal")

    # --- Road B Logic ---
    # Congestion Definition: y_B = 1 <-> f_B >= 1080
    model.addGenConstrIndicator(y_B, 1, f_B >= threshold_B, name="Congestion_Def_B_1")
    model.addGenConstrIndicator(y_B, 0, f_B <= threshold_B, name="Congestion_Def_B_0")
    
    # Travel Time Calculation
    # Base expression: 12 + 0.00008 * fB^2
    base_time_B_expr = params['travel_time_coefs']['B']['base'] + params['travel_time_coefs']['B']['coef'] * fB_sq
    
    # If y_B=1 (Congested): T_B = 1.05 * Base
    model.addGenConstrIndicator(y_B, 1, T_B == penalty_factor * base_time_B_expr, name="Time_B_Congested")
    # If y_B=0 (Normal): T_B = Base
    model.addGenConstrIndicator(y_B, 0, T_B == base_time_B_expr, name="Time_B_Normal")

    # 6. Set up the objective function.
    # Minimize Total Travel Time: Z = f_A * T_A + f_B * T_B
    # Note: f_A * T_A is a quadratic term of variables (since T_A is a variable). 
    # Gurobi NonConvex=2 handles this.
    model.setObjective(f_A * T_A + f_B * T_B, GRB.MINIMIZE)

    # 7. Solve the model and print results.
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("Optimization Successful")
        print(f"Allocated Flow A: {f_A.X}")
        print(f"Allocated Flow B: {f_B.X}")
        print(f"Travel Time A: {T_A.X}")
        print(f"Travel Time B: {T_B.X}")
        print(f"Objective Value: {model.ObjVal}")
        
        # Output the answer in the required format
        print(f"FinalAnswer=【{f_A.X}】")
    else:
        print("Optimization failed to find an optimal solution.")

if __name__ == "__main__":
    solve_traffic_problem()
import gurobipy as gp
from gurobipy import GRB


def solve_logistics_optimization(
    distance_packaging_loading=1400.0,
    daily_trips=[
        [0, 20, 15, 10],
        [20, 0, 25, 30],
        [15, 25, 0, 40],
        [10, 30, 40, 0]
    ],
    cost_per_meter_per_trip=0.50,
    minimal_distance=100.0
):
    """
    Solves the logistics optimization problem with nonlinear cost rules:
    - Extra fixed 3,000 CNY/day if Manhattan distance between Sorting and Packaging > 500 m.
    - If Manhattan distance between Storage and Loading < 250 m,
      cost per meter per trip between them becomes 0.80 instead of 0.50.
    """
    try:
        # --- Create a new model ---
        model = gp.Model("LogisticsOptimization")

        # --- Coordinates of fixed areas ---
        # Packaging at (0, 0)
        x_Pk, y_Pk = 0.0, 0.0
        # Loading at (800, 600)
        x_L, y_L = 800.0, 600.0

        # --- Decision Variables: coordinates of Sorting (S) and Storage (St) ---
        x_S = model.addVar(lb=0, ub=1000, name="x_Sorting")
        y_S = model.addVar(lb=0, ub=1000, name="y_Sorting")
        x_St = model.addVar(lb=0, ub=1000, name="x_Storage")
        y_St = model.addVar(lb=0, ub=1000, name="y_Storage")

        # --- Auxiliary distance variables (Manhattan distances) ---
        # Distances from Sorting
        d_S_Pk = model.addVar(lb=minimal_distance, name="Distance_Sorting_Packaging")
        d_S_L = model.addVar(lb=minimal_distance, name="Distance_Sorting_Loading")
        # Distances from Storage
        d_St_Pk = model.addVar(lb=minimal_distance, name="Distance_Storage_Packaging")
        d_St_L = model.addVar(lb=minimal_distance, name="Distance_Storage_Loading")
        # Distance between Sorting and Storage
        d_S_St = model.addVar(lb=minimal_distance, name="Distance_Sorting_Storage")
        # Fixed Packaging-Loading distance is constant
        d_Pk_L = distance_packaging_loading

        # Absolute value auxiliaries for Manhattan distances
        # Sorting - Packaging
        dx_S_Pk_pos = model.addVar(lb=0, name="dx_S_Pk_pos")
        dx_S_Pk_neg = model.addVar(lb=0, name="dx_S_Pk_neg")
        dy_S_Pk_pos = model.addVar(lb=0, name="dy_S_Pk_pos")
        dy_S_Pk_neg = model.addVar(lb=0, name="dy_S_Pk_neg")

        # Sorting - Loading
        dx_S_L_pos = model.addVar(lb=0, name="dx_S_L_pos")
        dx_S_L_neg = model.addVar(lb=0, name="dx_S_L_neg")
        dy_S_L_pos = model.addVar(lb=0, name="dy_S_L_pos")
        dy_S_L_neg = model.addVar(lb=0, name="dy_S_L_neg")

        # Storage - Packaging
        dx_St_Pk_pos = model.addVar(lb=0, name="dx_St_Pk_pos")
        dx_St_Pk_neg = model.addVar(lb=0, name="dx_St_Pk_neg")
        dy_St_Pk_pos = model.addVar(lb=0, name="dy_St_Pk_pos")
        dy_St_Pk_neg = model.addVar(lb=0, name="dy_St_Pk_neg")

        # Storage - Loading
        dx_St_L_pos = model.addVar(lb=0, name="dx_St_L_pos")
        dx_St_L_neg = model.addVar(lb=0, name="dx_St_L_neg")
        dy_St_L_pos = model.addVar(lb=0, name="dy_St_L_pos")
        dy_St_L_neg = model.addVar(lb=0, name="dy_St_L_neg")

        # Sorting - Storage
        dx_S_St_pos = model.addVar(lb=0, name="dx_S_St_pos")
        dx_S_St_neg = model.addVar(lb=0, name="dx_S_St_neg")
        dy_S_St_pos = model.addVar(lb=0, name="dy_S_St_pos")
        dy_S_St_neg = model.addVar(lb=0, name="dy_S_St_neg")

        # --- Model Manhattan distances via linear abs() representation ---
        # Sorting - Packaging: |x_S - 0| + |y_S - 0|
        model.addConstr(x_S - x_Pk == dx_S_Pk_pos - dx_S_Pk_neg, "c_dx_S_Pk")
        model.addConstr(y_S - y_Pk == dy_S_Pk_pos - dy_S_Pk_neg, "c_dy_S_Pk")
        model.addConstr(d_S_Pk == dx_S_Pk_pos + dx_S_Pk_neg + dy_S_Pk_pos + dy_S_Pk_neg, "c_d_S_Pk")

        # Sorting - Loading: |x_S - 800| + |y_S - 600|
        model.addConstr(x_S - x_L == dx_S_L_pos - dx_S_L_neg, "c_dx_S_L")
        model.addConstr(y_S - y_L == dy_S_L_pos - dy_S_L_neg, "c_dy_S_L")
        model.addConstr(d_S_L == dx_S_L_pos + dx_S_L_neg + dy_S_L_pos + dy_S_L_neg, "c_d_S_L")

        # Storage - Packaging: |x_St - 0| + |y_St - 0|
        model.addConstr(x_St - x_Pk == dx_St_Pk_pos - dx_St_Pk_neg, "c_dx_St_Pk")
        model.addConstr(y_St - y_Pk == dy_St_Pk_pos - dy_St_Pk_neg, "c_dy_St_Pk")
        model.addConstr(d_St_Pk == dx_St_Pk_pos + dx_St_Pk_neg + dy_St_Pk_pos + dy_St_Pk_neg, "c_d_St_Pk")

        # Storage - Loading: |x_St - 800| + |y_St - 600|
        model.addConstr(x_St - x_L == dx_St_L_pos - dx_St_L_neg, "c_dx_St_L")
        model.addConstr(y_St - y_L == dy_St_L_pos - dy_St_L_neg, "c_dy_St_L")
        model.addConstr(d_St_L == dx_St_L_pos + dx_St_L_neg + dy_St_L_pos + dy_St_L_neg, "c_d_St_L")

        # Sorting - Storage: |x_S - x_St| + |y_S - y_St|
        model.addConstr(x_S - x_St == dx_S_St_pos - dx_S_St_neg, "c_dx_S_St")
        model.addConstr(y_S - y_St == dy_S_St_pos - dy_S_St_neg, "c_dy_S_St")
        model.addConstr(d_S_St == dx_S_St_pos + dx_S_St_neg + dy_S_St_pos + dy_S_St_neg, "c_d_S_St")

        # --- Distance lower bounds between any two areas (already ensured via lb for S/St distances)
        # Fixed minimal distance 100 between all four areas:
        # Packaging-Sorting, Packaging-Storage, Loading-Sorting, Loading-Storage,
        # Sorting-Storage are already >= minimal_distance via variable bounds.

        # --- ❤ Non-linearity is introduced. ❤
        # --- Original linear objective portion is commented out below ---
        # cost_Pk_L = (daily_trips[0][1] + daily_trips[1][0]) * distance_packaging_loading
        # cost_Pk_S = (daily_trips[0][2] + daily_trips[2][0]) * d_S_Pk
        # cost_Pk_St = (daily_trips[0][3] + daily_trips[3][0]) * d_St_Pk
        # cost_L_S = (daily_trips[1][2] + daily_trips[2][1]) * d_S_L
        # cost_L_St = (daily_trips[1][3] + daily_trips[3][1]) * d_St_L
        # cost_S_St = (daily_trips[2][3] + daily_trips[3][2]) * d_S_St
        # total_transport_cost_before_factor = (
        #     cost_Pk_L + cost_Pk_S + cost_Pk_St + cost_L_S + cost_L_St + cost_S_St
        # )
        # model.setObjective(total_transport_cost_before_factor * cost_per_meter_per_trip, GRB.MINIMIZE)

        # --- Binary variables for nonlinear cost rules (implemented as MILP) ---
        # 1) Extra 3000 CNY/day if d_S_Pk > 500
        z_S_Pk_long = model.addVar(vtype=GRB.BINARY, name="z_S_Pk_long")

        # 2) Higher variable cost (0.80) if d_St_L < 250
        z_St_L_close = model.addVar(vtype=GRB.BINARY, name="z_St_L_close")

        # --- Big-M constants ---
        M_dist = 2000.0  # upper bound on Manhattan distance inside 1000x1000 (safe)

        # --- Logic for z_S_Pk_long: d_S_Pk > 500 triggers fixed 3000 cost ---
        # z_S_Pk_long = 1  =>  d_S_Pk >= 500
        model.addConstr(d_S_Pk >= 500 - M_dist * (1 - z_S_Pk_long), "c_S_Pk_long_lb")
        # z_S_Pk_long = 0  =>  d_S_Pk <= 500
        model.addConstr(d_S_Pk <= 500 + M_dist * z_S_Pk_long, "c_S_Pk_long_ub")

        # --- Logic for z_St_L_close: d_St_L < 250 gives higher cost ---
        # z_St_L_close = 1  =>  d_St_L <= 250
        model.addConstr(d_St_L <= 250 + M_dist * (1 - z_St_L_close), "c_St_L_close_ub")
        # z_St_L_close = 0  =>  d_St_L >= 250
        model.addConstr(d_St_L >= 250 - M_dist * z_St_L_close, "c_St_L_close_lb")

        # --- Objective Function with nonlinear (piecewise) rules encoded linearly ---

        # Symmetric daily trips (round trips) between pairs:
        trips_Pk_L = daily_trips[0][1] + daily_trips[1][0]  # 20 + 20
        trips_Pk_S = daily_trips[0][2] + daily_trips[2][0]  # 15 + 15
        trips_Pk_St = daily_trips[0][3] + daily_trips[3][0]  # 10 + 10
        trips_L_S = daily_trips[1][2] + daily_trips[2][1]  # 25 + 25
        trips_L_St = daily_trips[1][3] + daily_trips[3][1]  # 30 + 30
        trips_S_St = daily_trips[2][3] + daily_trips[3][2]  # 40 + 40

        # Base per-meter cost for all flows except Storage-Loading (may be overridden)
        c_base = cost_per_meter_per_trip  # 0.50

        # 1) Packaging - Loading (fixed distance, base cost)
        cost_Pk_L = trips_Pk_L * d_Pk_L * c_base

        # 2) Packaging - Sorting (distance d_S_Pk, base cost)
        cost_Pk_S = trips_Pk_S * d_S_Pk * c_base

        # 3) Packaging - Storage (distance d_St_Pk, base cost)
        cost_Pk_St = trips_Pk_St * d_St_Pk * c_base

        # 4) Loading - Sorting (distance d_S_L, base cost)
        cost_L_S = trips_L_S * d_S_L * c_base

        # 5) Sorting - Storage (distance d_S_St, base cost)
        cost_S_St = trips_S_St * d_S_St * c_base

        # 6) Storage - Loading:
        #    cost per meter is:
        #       0.80 if d_St_L < 250 (z_St_L_close = 1)
        #       0.50 if d_St_L >= 250 (z_St_L_close = 0)
        #    We linearize as:
        #       cost_L_St = trips_L_St * d_St_L * (0.50 + (0.80 - 0.50) * z_St_L_close)
        #                 = trips_L_St * d_St_L * (0.50 + 0.30 * z_St_L_close)
        higher_cost_increment = 0.80 - c_base  # 0.30
        effective_cost_St_L = c_base + higher_cost_increment * z_St_L_close
        cost_L_St = trips_L_St * d_St_L * effective_cost_St_L

        # 7) Fixed 3000 CNY/day if d_S_Pk > 500 (z_S_Pk_long = 1)
        fixed_extra_cost_S_Pk = 3000.0 * z_S_Pk_long

        total_cost = (
            cost_Pk_L
            + cost_Pk_S
            + cost_Pk_St
            + cost_L_S
            + cost_L_St
            + cost_S_St
            + fixed_extra_cost_S_Pk
        )

        model.setObjective(total_cost, GRB.MINIMIZE)

        # --- Optimize model ---
        model.optimize()

        # --- Return solution ---
        if model.status == GRB.OPTIMAL:
            solution = {
                "status": "optimal",
                "obj": model.ObjVal,
                "x_S": x_S.X,
                "y_S": y_S.X,
                "x_St": x_St.X,
                "y_St": y_St.X,
                "d_S_Pk": d_S_Pk.X,
                "d_S_L": d_S_L.X,
                "d_St_Pk": d_St_Pk.X,
                "d_St_L": d_St_L.X,
                "d_S_St": d_S_St.X,
                "z_S_Pk_long": z_S_Pk_long.X,
                "z_St_L_close": z_St_L_close.X,
            }
            return solution
        else:
            return {"status": f"{model.status}"}

    except gp.GurobiError as e:
        return {"status": f"Gurobi error: {e}"}
    except Exception as e:
        return {"status": f"An unexpected error occurred: {e}"}


if __name__ == "__main__":
    result = solve_logistics_optimization()
    print(result)
import gurobipy as gp
from gurobipy import GRB
import math


def solve_logistics_optimization():
    distance_packaging_loading = 1400.0
    daily_trips = [
        [0, 20, 15, 10],
        [20, 0, 25, 30],
        [15, 25, 0, 40],
        [10, 30, 40, 0]
    ]
    cost_per_meter_per_trip = 0.50
    minimal_distance = 100.0
    growth_factor = 1.3
    """
    Solves the logistics optimization problem with a non-linear, exponentially
    increasing effective distance cost for safety spacing.

    NOTE: The 'distance_packaging_loading' parameter is kept for backward
    compatibility with the original interface but is no longer used directly
    to define distances, since we now model coordinates explicitly.
    """
    try:
        # --- Create a new model ---
        model = gp.Model("LogisticsOptimization_Nonlinear")

        # ------------------------------------------------------------------
        # Coordinate variables (Manhattan movement in a 1000x1000 rectangle)
        # ------------------------------------------------------------------
        # Packaging (Pk) is fixed at (0, 0)
        x_Pk, y_Pk = 0.0, 0.0

        # Loading (L) is fixed at (800, 600)
        x_L, y_L = 800.0, 600.0

        # Sorting facility (S) coordinates (decision variables)
        x_S = model.addVar(lb=0.0, ub=1000.0, name="x_Sorting")
        y_S = model.addVar(lb=0.0, ub=1000.0, name="y_Sorting")

        # Storage facility (St) coordinates (decision variables)
        x_St = model.addVar(lb=0.0, ub=1000.0, name="x_Storage")
        y_St = model.addVar(lb=0.0, ub=1000.0, name="y_Storage")

        # ------------------------------------------------------------------
        # Manhattan distance variables between all pairs
        # ------------------------------------------------------------------
        # Distances: Pk-L, Pk-S, Pk-St, L-S, L-St, S-St
        d_Pk_L = model.addVar(lb=minimal_distance, name="d_Pk_L")
        d_Pk_S = model.addVar(lb=minimal_distance, name="d_Pk_S")
        d_Pk_St = model.addVar(lb=minimal_distance, name="d_Pk_St")
        d_L_S = model.addVar(lb=minimal_distance, name="d_L_S")
        d_L_St = model.addVar(lb=minimal_distance, name="d_L_St")
        d_S_St = model.addVar(lb=minimal_distance, name="d_S_St")

        # Auxiliary variables for absolute values (Pk-L has constant coords)
        # Pk-S
        dx_Pk_S = model.addVar(lb=-GRB.INFINITY, name="dx_Pk_S")
        dy_Pk_S = model.addVar(lb=-GRB.INFINITY, name="dy_Pk_S")
        abs_dx_Pk_S = model.addVar(lb=0.0, name="abs_dx_Pk_S")
        abs_dy_Pk_S = model.addVar(lb=0.0, name="abs_dy_Pk_S")

        # Pk-St
        dx_Pk_St = model.addVar(lb=-GRB.INFINITY, name="dx_Pk_St")
        dy_Pk_St = model.addVar(lb=-GRB.INFINITY, name="dy_Pk_St")
        abs_dx_Pk_St = model.addVar(lb=0.0, name="abs_dx_Pk_St")
        abs_dy_Pk_St = model.addVar(lb=0.0, name="abs_dy_Pk_St")

        # L-S
        dx_L_S = model.addVar(lb=-GRB.INFINITY, name="dx_L_S")
        dy_L_S = model.addVar(lb=-GRB.INFINITY, name="dy_L_S")
        abs_dx_L_S = model.addVar(lb=0.0, name="abs_dx_L_S")
        abs_dy_L_S = model.addVar(lb=0.0, name="abs_dy_L_S")

        # L-St
        dx_L_St = model.addVar(lb=-GRB.INFINITY, name="dx_L_St")
        dy_L_St = model.addVar(lb=-GRB.INFINITY, name="dy_L_St")
        abs_dx_L_St = model.addVar(lb=0.0, name="abs_dx_L_St")
        abs_dy_L_St = model.addVar(lb=0.0, name="abs_dy_L_St")

        # S-St
        dx_S_St = model.addVar(lb=-GRB.INFINITY, name="dx_S_St")
        dy_S_St = model.addVar(lb=-GRB.INFINITY, name="dy_S_St")
        abs_dx_S_St = model.addVar(lb=0.0, name="abs_dx_S_St")
        abs_dy_S_St = model.addVar(lb=0.0, name="abs_dy_S_St")

        # ------------------------------------------------------------------
        # Manhattan distance constraints
        # d(i,j) = |x_i - x_j| + |y_i - y_j|
        # Absolute values are linearized using standard technique
        # ------------------------------------------------------------------

        # Fixed Pk-L distance from coordinates (0,0) and (800,600)
        model.addConstr(d_Pk_L >= abs(800.0) + abs(600.0), name="d_Pk_L_lb")
        model.addConstr(d_Pk_L == 800.0 + 600.0, name="d_Pk_L_fix")

        # Pk-S: origin is (0,0), so dx, dy are just x_S, y_S
        model.addConstr(dx_Pk_S == x_S - x_Pk, name="dx_Pk_S_def")
        model.addConstr(dy_Pk_S == y_S - y_Pk, name="dy_Pk_S_def")

        model.addConstr(abs_dx_Pk_S >= dx_Pk_S, name="abs_dx_Pk_S_pos")
        model.addConstr(abs_dx_Pk_S >= -dx_Pk_S, name="abs_dx_Pk_S_neg")
        model.addConstr(abs_dy_Pk_S >= dy_Pk_S, name="abs_dy_Pk_S_pos")
        model.addConstr(abs_dy_Pk_S >= -dy_Pk_S, name="abs_dy_Pk_S_neg")
        model.addConstr(d_Pk_S == abs_dx_Pk_S + abs_dy_Pk_S, name="d_Pk_S_def")

        # Pk-St
        model.addConstr(dx_Pk_St == x_St - x_Pk, name="dx_Pk_St_def")
        model.addConstr(dy_Pk_St == y_St - y_Pk, name="dy_Pk_St_def")

        model.addConstr(abs_dx_Pk_St >= dx_Pk_St, name="abs_dx_Pk_St_pos")
        model.addConstr(abs_dx_Pk_St >= -dx_Pk_St, name="abs_dx_Pk_St_neg")
        model.addConstr(abs_dy_Pk_St >= dy_Pk_St, name="abs_dy_Pk_St_pos")
        model.addConstr(abs_dy_Pk_St >= -dy_Pk_St, name="abs_dy_Pk_St_neg")
        model.addConstr(d_Pk_St == abs_dx_Pk_St + abs_dy_Pk_St, name="d_Pk_St_def")

        # L-S
        model.addConstr(dx_L_S == x_S - x_L, name="dx_L_S_def")
        model.addConstr(dy_L_S == y_S - y_L, name="dy_L_S_def")

        model.addConstr(abs_dx_L_S >= dx_L_S, name="abs_dx_L_S_pos")
        model.addConstr(abs_dx_L_S >= -dx_L_S, name="abs_dx_L_S_neg")
        model.addConstr(abs_dy_L_S >= dy_L_S, name="abs_dy_L_S_pos")
        model.addConstr(abs_dy_L_S >= -dy_L_S, name="abs_dy_L_S_neg")
        model.addConstr(d_L_S == abs_dx_L_S + abs_dy_L_S, name="d_L_S_def")

        # L-St
        model.addConstr(dx_L_St == x_St - x_L, name="dx_L_St_def")
        model.addConstr(dy_L_St == y_St - y_L, name="dy_L_St_def")

        model.addConstr(abs_dx_L_St >= dx_L_St, name="abs_dx_L_St_pos")
        model.addConstr(abs_dx_L_St >= -dx_L_St, name="abs_dx_L_St_neg")
        model.addConstr(abs_dy_L_St >= dy_L_St, name="abs_dy_L_St_pos")
        model.addConstr(abs_dy_L_St >= -dy_L_St, name="abs_dy_L_St_neg")
        model.addConstr(d_L_St == abs_dx_L_St + abs_dy_L_St, name="d_L_St_def")

        # S-St
        model.addConstr(dx_S_St == x_S - x_St, name="dx_S_St_def")
        model.addConstr(dy_S_St == y_S - y_St, name="dy_S_St_def")

        model.addConstr(abs_dx_S_St >= dx_S_St, name="abs_dx_S_St_pos")
        model.addConstr(abs_dx_S_St >= -dx_S_St, name="abs_dx_S_St_neg")
        model.addConstr(abs_dy_S_St >= dy_S_St, name="abs_dy_S_St_pos")
        model.addConstr(abs_dy_S_St >= -dy_S_St, name="abs_dy_S_St_neg")
        model.addConstr(d_S_St == abs_dx_S_St + abs_dy_S_St, name="d_S_St_def")

        # ------------------------------------------------------------------
        # Minimum distance constraints between every pair (>= minimal_distance)
        # (Already set as lb when defining distance variables)
        # ------------------------------------------------------------------
        # Still, add explicit constraints to enforce clarity and robustness.
        model.addConstr(d_Pk_L >= minimal_distance, name="min_dist_Pk_L")
        model.addConstr(d_Pk_S >= minimal_distance, name="min_dist_Pk_S")
        model.addConstr(d_Pk_St >= minimal_distance, name="min_dist_Pk_St")
        model.addConstr(d_L_S >= minimal_distance, name="min_dist_L_S")
        model.addConstr(d_L_St >= minimal_distance, name="min_dist_L_St")
        model.addConstr(d_S_St >= minimal_distance, name="min_dist_S_St")

        # ------------------------------------------------------------------
        # Non-linear "effective distance" definition:
        #   d_eff = d * (1.3^n),  n = ceil(d / 100)
        # Implemented using a piecewise-constant exponent (stepwise) approximation
        # over the feasible range of Manhattan distances.
        # ------------------------------------------------------------------

        # Upper bound on Manhattan distance in the 1000x1000 grid:
        # Max distance between (0,0) and (1000,1000) is 2000.
        d_max = 2000.0

        # We build breakpoints for integer layers of n = ceil(d / 100)
        # For each layer k, for distances in ((k-1)*100, k*100], the exponent is k.
        # We model d_eff = d * (growth_factor^k) piecewise w.r.t d.
        #
        # To make this Gurobi-friendly and keep it convex (growth_factor>1),
        # we use a piecewise linear approximation of the function
        # f(d) = d * growth_factor^ceil(d/100) on [minimal_distance, d_max].
        #
        # For simplicity, we use a dense grid of breakpoints at every 100 units.

        def build_piecewise_effective_distance(d_var, name_prefix):
            # Breakpoints for d
            pts = []
            vals = []
            k_min = int(math.ceil(minimal_distance / 100.0))
            k_max = int(math.ceil(d_max / 100.0))

            # Construct segments: for each layer k, we sample at d = (k-1)*100 and d = k*100
            for k in range(k_min, k_max + 1):
                left = max(minimal_distance, (k - 1) * 100.0)
                right = k * 100.0
                if pts and abs(left - pts[-1]) < 1e-6:
                    # Avoid duplicate point
                    pass
                else:
                    pts.append(left)
                    n_left = int(math.ceil(left / 100.0))
                    vals.append(left * (growth_factor ** n_left))
                if right <= d_max:
                    pts.append(right)
                    n_right = int(math.ceil(right / 100.0))
                    vals.append(right * (growth_factor ** n_right))

            # Ensure strictly increasing pts
            # (Tiny perturbations could be added if needed, but here they are exact multiples)
            d_eff = model.addVar(lb=0.0, name=f"{name_prefix}_effective")

            # ❤ Non-linearity is introduced. ❤
            model.addGenConstrPWL(d_var, d_eff, pts, vals, name=f"{name_prefix}_PWL")

            return d_eff

        # Effective distances for each pair
        d_eff_Pk_L = build_piecewise_effective_distance(d_Pk_L, "d_Pk_L")
        d_eff_Pk_S = build_piecewise_effective_distance(d_Pk_S, "d_Pk_S")
        d_eff_Pk_St = build_piecewise_effective_distance(d_Pk_St, "d_Pk_St")
        d_eff_L_S = build_piecewise_effective_distance(d_L_S, "d_L_S")
        d_eff_L_St = build_piecewise_effective_distance(d_L_St, "d_L_St")
        d_eff_S_St = build_piecewise_effective_distance(d_S_St, "d_S_St")

        # ------------------------------------------------------------------
        # Objective function
        # Replace original linear distance cost with effective (nonlinear) costs
        # ------------------------------------------------------------------

        # Daily round-trip flows between zones:
        # indices: 0-Pk, 1-L, 2-S, 3-St

        # Original linear-cost definition (commented out):
        # ❤ Non-linearity is introduced. ❤
        # d_S_St = model.addVar(lb=minimal_distance, name="DistanceBetweenSortingAndStorage")
        # d_S_Pk = model.addVar(lb=minimal_distance, name="DistanceBetweenSortingAndPackaging")
        # d_S_L = model.addVar(lb=minimal_distance, name="DistanceBetweenSortingAndLoading")
        # d_St_Pk = model.addVar(lb=minimal_distance, name="DistanceBetweenStorageAndPackaging")
        # d_St_L = model.addVar(lb=minimal_distance, name="DistanceBetweenStorageAndLoading")
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

        # Now use effective distances
        flow_Pk_L = daily_trips[0][1] + daily_trips[1][0]
        flow_Pk_S = daily_trips[0][2] + daily_trips[2][0]
        flow_Pk_St = daily_trips[0][3] + daily_trips[3][0]
        flow_L_S = daily_trips[1][2] + daily_trips[2][1]
        flow_L_St = daily_trips[1][3] + daily_trips[3][1]
        flow_S_St = daily_trips[2][3] + daily_trips[3][2]

        cost_Pk_L = flow_Pk_L * d_eff_Pk_L
        cost_Pk_S = flow_Pk_S * d_eff_Pk_S
        cost_Pk_St = flow_Pk_St * d_eff_Pk_St
        cost_L_S = flow_L_S * d_eff_L_S
        cost_L_St = flow_L_St * d_eff_L_St
        cost_S_St = flow_S_St * d_eff_S_St

        total_effective_cost = (
            cost_Pk_L
            + cost_Pk_S
            + cost_Pk_St
            + cost_L_S
            + cost_L_St
            + cost_S_St
        )

        model.setObjective(total_effective_cost * cost_per_meter_per_trip, GRB.MINIMIZE)

        # ------------------------------------------------------------------
        # Old collinearity / max distance constraints (no longer applicable)
        # Commented out and tagged as where non-linearity was introduced.
        # ------------------------------------------------------------------
        # ❤ Non-linearity is introduced. ❤
        # model.addConstr(d_S_L + d_S_Pk == distance_packaging_loading, "Collinearity_Sorting")
        # model.addConstr(d_St_Pk + d_St_L == distance_packaging_loading, "Collinearity_Storage")
        # model.addConstr(d_S_St <= distance_packaging_loading, "MaxDist_SortingStorage")

        # ------------------------------------------------------------------
        # Optimize model
        # ------------------------------------------------------------------
        model.optimize()

        # ------------------------------------------------------------------
        # Return solution
        # ------------------------------------------------------------------
        if model.status == GRB.OPTIMAL:
            solution = {
                "status": "optimal",
                "obj": model.ObjVal,
                "sorting_coordinates": (x_S.X, y_S.X),
                "storage_coordinates": (x_St.X, y_St.X),
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
import gurobipy as gp
from gurobipy import GRB
import math


def solve_logistics_optimization(
    cost_per_meter_per_trip=0.50,
    manhattan_euclid_ratio_min=1.2
):
    """
    Solves the logistics optimization problem with:
    - Packaging at (0, 0)
    - Loading at (800, 600)
    - Sorting (S) and Storage (St) to be located in [0,1000]x[0,1000]
    - All movement costed using Manhattan distance (axis-aligned)
    - Non-linear constraint: (sum of all pairwise Manhattan distances) /
                             (sum of all pairwise Euclidean distances) >= 1.2
    """

    try:
        # --- Create a new model ---
        model = gp.Model("LogisticsOptimization_Nonlinear")

        # --- Known coordinates ---
        # Packaging (Pk) at (0, 0)
        x_Pk, y_Pk = 0.0, 0.0
        # Loading (L) at (800, 600)
        x_L, y_L = 800.0, 600.0

        # --- Decision Variables: coordinates of Sorting (S) and Storage (St) ---
        x_S = model.addVar(lb=0.0, ub=1000.0, name="x_Sorting")
        y_S = model.addVar(lb=0.0, ub=1000.0, name="y_Sorting")
        x_St = model.addVar(lb=0.0, ub=1000.0, name="x_Storage")
        y_St = model.addVar(lb=0.0, ub=1000.0, name="y_Storage")

        # --- Auxiliary variables for pairwise Manhattan distances ---
        # |x_i - x_j| and |y_i - y_j| for all pairs (i,j)
        abs_x_Pk_L = model.addVar(lb=0.0, name="abs_x_Pk_L")
        abs_y_Pk_L = model.addVar(lb=0.0, name="abs_y_Pk_L")

        abs_x_Pk_S = model.addVar(lb=0.0, name="abs_x_Pk_S")
        abs_y_Pk_S = model.addVar(lb=0.0, name="abs_y_Pk_S")

        abs_x_Pk_St = model.addVar(lb=0.0, name="abs_x_Pk_St")
        abs_y_Pk_St = model.addVar(lb=0.0, name="abs_y_Pk_St")

        abs_x_L_S = model.addVar(lb=0.0, name="abs_x_L_S")
        abs_y_L_S = model.addVar(lb=0.0, name="abs_y_L_S")

        abs_x_L_St = model.addVar(lb=0.0, name="abs_x_L_St")
        abs_y_L_St = model.addVar(lb=0.0, name="abs_y_L_St")

        abs_x_S_St = model.addVar(lb=0.0, name="abs_x_S_St")
        abs_y_S_St = model.addVar(lb=0.0, name="abs_y_S_St")

        # Manhattan distances d_M_ij = |x_i - x_j| + |y_i - y_j|
        dM_Pk_L = model.addVar(lb=0.0, name="dM_Pk_L")
        dM_Pk_S = model.addVar(lb=0.0, name="dM_Pk_S")
        dM_Pk_St = model.addVar(lb=0.0, name="dM_Pk_St")
        dM_L_S = model.addVar(lb=0.0, name="dM_L_S")
        dM_L_St = model.addVar(lb=0.0, name="dM_L_St")
        dM_S_St = model.addVar(lb=0.0, name="dM_S_St")

        # --- Auxiliary variables for pairwise Euclidean distances ---
        # dE_ij = sqrt( (x_i - x_j)^2 + (y_i - y_j)^2 )
        dE_Pk_L = model.addVar(lb=0.0, name="dE_Pk_L")
        dE_Pk_S = model.addVar(lb=0.0, name="dE_Pk_S")
        dE_Pk_St = model.addVar(lb=0.0, name="dE_Pk_St")
        dE_L_S = model.addVar(lb=0.0, name="dE_L_S")
        dE_L_St = model.addVar(lb=0.0, name="dE_L_St")
        dE_S_St = model.addVar(lb=0.0, name="dE_S_St")

        # --- Daily trips matrix (same as original) ---
        daily_trips = [
            [0, 20, 15, 10],
            [20, 0, 25, 30],
            [15, 25, 0, 40],
            [10, 30, 40, 0]
        ]
        # Index mapping: 0=Pk, 1=L, 2=S, 3=St

        # --- Linear constraints to model absolute values for Manhattan distances ---

        # Pk-L (coordinates fixed, distances known, but keep generic)
        # x_Pk = 0, y_Pk = 0; x_L = 800, y_L = 600 (constants)
        model.addConstr(abs_x_Pk_L >= x_Pk - x_L, "abs_x_Pk_L_pos")
        model.addConstr(abs_x_Pk_L >= x_L - x_Pk, "abs_x_Pk_L_neg")
        model.addConstr(abs_y_Pk_L >= y_Pk - y_L, "abs_y_Pk_L_pos")
        model.addConstr(abs_y_Pk_L >= y_L - y_Pk, "abs_y_Pk_L_neg")
        model.addConstr(dM_Pk_L == abs_x_Pk_L + abs_y_Pk_L, "dM_Pk_L_def")

        # Pk-S
        model.addConstr(abs_x_Pk_S >= x_Pk - x_S, "abs_x_Pk_S_pos")
        model.addConstr(abs_x_Pk_S >= x_S - x_Pk, "abs_x_Pk_S_neg")
        model.addConstr(abs_y_Pk_S >= y_Pk - y_S, "abs_y_Pk_S_pos")
        model.addConstr(abs_y_Pk_S >= y_S - y_Pk, "abs_y_Pk_S_neg")
        model.addConstr(dM_Pk_S == abs_x_Pk_S + abs_y_Pk_S, "dM_Pk_S_def")

        # Pk-St
        model.addConstr(abs_x_Pk_St >= x_Pk - x_St, "abs_x_Pk_St_pos")
        model.addConstr(abs_x_Pk_St >= x_St - x_Pk, "abs_x_Pk_St_neg")
        model.addConstr(abs_y_Pk_St >= y_Pk - y_St, "abs_y_Pk_St_pos")
        model.addConstr(abs_y_Pk_St >= y_St - y_Pk, "abs_y_Pk_St_neg")
        model.addConstr(dM_Pk_St == abs_x_Pk_St + abs_y_Pk_St, "dM_Pk_St_def")

        # L-S
        model.addConstr(abs_x_L_S >= x_L - x_S, "abs_x_L_S_pos")
        model.addConstr(abs_x_L_S >= x_S - x_L, "abs_x_L_S_neg")
        model.addConstr(abs_y_L_S >= y_L - y_S, "abs_y_L_S_pos")
        model.addConstr(abs_y_L_S >= y_S - y_L, "abs_y_L_S_neg")
        model.addConstr(dM_L_S == abs_x_L_S + abs_y_L_S, "dM_L_S_def")

        # L-St
        model.addConstr(abs_x_L_St >= x_L - x_St, "abs_x_L_St_pos")
        model.addConstr(abs_x_L_St >= x_St - x_L, "abs_x_L_St_neg")
        model.addConstr(abs_y_L_St >= y_L - y_St, "abs_y_L_St_pos")
        model.addConstr(abs_y_L_St >= y_St - y_L, "abs_y_L_St_neg")
        model.addConstr(dM_L_St == abs_x_L_St + abs_y_L_St, "dM_L_St_def")

        # S-St
        model.addConstr(abs_x_S_St >= x_S - x_St, "abs_x_S_St_pos")
        model.addConstr(abs_x_S_St >= x_St - x_S, "abs_x_S_St_neg")
        model.addConstr(abs_y_S_St >= y_S - y_St, "abs_y_S_St_pos")
        model.addConstr(abs_y_S_St >= y_St - y_S, "abs_y_S_St_neg")
        model.addConstr(dM_S_St == abs_x_S_St + abs_y_S_St, "dM_S_St_def")

        # --- Euclidean distances via quadratic constraints ---
        # dE_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2

        # Pk-L (constants)
        dx_Pk_L = x_Pk - x_L
        dy_Pk_L = y_Pk - y_L
        model.addQConstr(dE_Pk_L * dE_Pk_L ==
                         dx_Pk_L * dx_Pk_L + dy_Pk_L * dy_Pk_L,
                         "dE_Pk_L_def")

        # Pk-S
        model.addQConstr(dE_Pk_S * dE_Pk_S ==
                         (x_Pk - x_S) * (x_Pk - x_S) +
                         (y_Pk - y_S) * (y_Pk - y_S),
                         "dE_Pk_S_def")

        # Pk-St
        model.addQConstr(dE_Pk_St * dE_Pk_St ==
                         (x_Pk - x_St) * (x_Pk - x_St) +
                         (y_Pk - y_St) * (y_Pk - y_St),
                         "dE_Pk_St_def")

        # L-S
        model.addQConstr(dE_L_S * dE_L_S ==
                         (x_L - x_S) * (x_L - x_S) +
                         (y_L - y_S) * (y_L - y_S),
                         "dE_L_S_def")

        # L-St
        model.addQConstr(dE_L_St * dE_L_St ==
                         (x_L - x_St) * (x_L - x_St) +
                         (y_L - y_St) * (y_L - y_St),
                         "dE_L_St_def")

        # S-St
        model.addQConstr(dE_S_St * dE_S_St ==
                         (x_S - x_St) * (x_S - x_St) +
                         (y_S - y_St) * (y_S - y_St),
                         "dE_S_St_def")

        # --- Transportation cost (using Manhattan distances) ---

        # Total daily trips between pairs (both directions)
        trips_Pk_L = daily_trips[0][1] + daily_trips[1][0]
        trips_Pk_S = daily_trips[0][2] + daily_trips[2][0]
        trips_Pk_St = daily_trips[0][3] + daily_trips[3][0]
        trips_L_S = daily_trips[1][2] + daily_trips[2][1]
        trips_L_St = daily_trips[1][3] + daily_trips[3][1]
        trips_S_St = daily_trips[2][3] + daily_trips[3][2]

        # Cost based on Manhattan distances
        cost_Pk_L = trips_Pk_L * dM_Pk_L
        cost_Pk_S = trips_Pk_S * dM_Pk_S
        cost_Pk_St = trips_Pk_St * dM_Pk_St
        cost_L_S = trips_L_S * dM_L_S
        cost_L_St = trips_L_St * dM_L_St
        cost_S_St = trips_S_St * dM_S_St

        total_transport_cost_before_factor = (
            cost_Pk_L
            + cost_Pk_S
            + cost_Pk_St
            + cost_L_S
            + cost_L_St
            + cost_S_St
        )

        model.setObjective(
            total_transport_cost_before_factor * cost_per_meter_per_trip,
            GRB.MINIMIZE,
        )

        # --- Non-linear Manhattan/Euclidean ratio constraint ---

        # Sums of Manhattan and Euclidean distances over all 6 unordered pairs
        sum_manhattan = (
            dM_Pk_L
            + dM_Pk_S
            + dM_Pk_St
            + dM_L_S
            + dM_L_St
            + dM_S_St
        )
        sum_euclid = (
            dE_Pk_L
            + dE_Pk_S
            + dE_Pk_St
            + dE_L_S
            + dE_L_St
            + dE_S_St
        )

        # ❤ Non-linearity is introduced. ❤
        # model.addConstr(sum_manhattan >= manhattan_euclid_ratio_min * sum_euclid,
        #                 "Manhattan_Euclid_Ratio")

        # Introduce the non-linear ratio constraint using a quadratic form:
        # sum_manhattan >= 1.2 * sum_euclid
        # This is directly supported as a general constraint in Gurobi's
        # non-linear (quadratic) framework.
        model.addConstr(
            sum_manhattan >= manhattan_euclid_ratio_min * sum_euclid,
            "Manhattan_Euclid_Ratio_Nonlinear",
        )

        # --- Optimize model ---
        model.Params.NonConvex = 2  # allow non-convex quadratic constraints
        model.optimize()

        # --- Return solution ---
        if model.status == GRB.OPTIMAL:
            return {
                "status": "optimal",
                "obj": model.ObjVal,
                "x_S": x_S.X,
                "y_S": y_S.X,
                "x_St": x_St.X,
                "y_St": y_St.X,
            }
        else:
            return {"status": f"{model.status}"}

    except gp.GurobiError as e:
        return {"status": f"Gurobi error: {e}"}
    except Exception as e:
        return {"status": f"An unexpected error occurred: {e}"}


if __name__ == "__main__":
    result = solve_logistics_optimization()
    print(result)
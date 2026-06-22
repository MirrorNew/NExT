import math
import gurobipy as gp
from gurobipy import GRB


def solve_candy_production():
    """
    Solves the candy factory production planning problem to maximize profit
    for a given month t, considering a cosine-based seasonal upper bound
    on effective demand for Candy A.

    Parameters
    ----------
    t : int or float
        Month index (t = 1, 2, 3, ...). Default is 3.
    """
    try:
        # --- Data ---
        candy_brands = ['CandyA', 'CandyB', 'CandyC']
        raw_materials = ['RawA', 'RawB', 'RawC']

        # Selling prices (Yuan/kg)
        selling_prices = {'CandyA': 3.40, 'CandyB': 2.85, 'CandyC': 2.25}

        # Processing fees (Yuan/kg)
        processing_fees = {'CandyA': 0.50, 'CandyB': 0.40, 'CandyC': 0.30}

        # Raw material costs (Yuan/kg)
        raw_material_costs = {'RawA': 2.00, 'RawB': 1.50, 'RawC': 1.00}

        # Monthly limits of raw materials (kg)
        raw_material_limits = {'RawA': 2000, 'RawB': 2500, 'RawC': 1200}

        # --- Create Gurobi Model ---
        # ❤ Non-linearity is introduced. ❤
        # model = gp.Model("CandyFactoryOptimization")
        model = gp.Model("CandyFactoryOptimization_Nonlinear")

        # --- Decision Variables ---
        # X[j]: total kilograms of candy brand j produced
        X = model.addVars(candy_brands,
                          name="X_prod",
                          lb=0.0,
                          vtype=GRB.CONTINUOUS)

        # R[i,j]: kilograms of raw material i used in candy brand j
        R = model.addVars(raw_materials,
                          candy_brands,
                          name="R_raw_in_candy",
                          lb=0.0,
                          vtype=GRB.CONTINUOUS)

        # --- Objective Function: Maximize Total Profit ---
        # Profit = (Sales Revenue - Processing Cost) - Raw Material Cost

        revenue_minus_processing = gp.quicksum(
            (selling_prices[j] - processing_fees[j]) * X[j]
            for j in candy_brands)

        total_raw_material_cost = gp.quicksum(raw_material_costs[i] * R[i, j]
                                              for i in raw_materials
                                              for j in candy_brands)

        model.setObjective(revenue_minus_processing - total_raw_material_cost,
                           GRB.MAXIMIZE)

        # --- Constraints ---
        # 1. Mass Balance for Each Candy Brand: Sum of raw materials = Total candy produced
        for j in candy_brands:
            model.addConstr(gp.quicksum(R[i, j]
                                        for i in raw_materials) == X[j],
                            name=f"MassBalance_{j}")

        # 2. Raw Material Availability Limits
        for i in raw_materials:
            model.addConstr(gp.quicksum(R[i, j] for j in candy_brands)
                            <= raw_material_limits[i],
                            name=f"Limit_{i}")

        # 3. Content Percentage Requirements
        # Candy A
        model.addConstr(R['RawA', 'CandyA'] >= 0.60 * X['CandyA'],
                        name="Content_CandyA_RawA_min")
        model.addConstr(R['RawC', 'CandyA'] <= 0.20 * X['CandyA'],
                        name="Content_CandyA_RawC_max")

        # Candy B
        model.addConstr(R['RawA', 'CandyB'] >= 0.15 * X['CandyB'],
                        name="Content_CandyB_RawA_min")
        model.addConstr(R['RawC', 'CandyB'] <= 0.60 * X['CandyB'],
                        name="Content_CandyB_RawC_max")

        # Candy C
        model.addConstr(R['RawC', 'CandyC'] <= 0.50 * X['CandyC'],
                        name="Content_CandyC_RawC_max")

        t = model.addVar(vtype=GRB.INTEGER,ub=12)
        Y_t = model.addVar(vtype=GRB.CONTINUOUS)
        cos_Y_T = model.addVar(vtype=GRB.CONTINUOUS)
        # 4. Seasonal effective demand upper bound for Candy A in month t
        #    x_A * [1 + 0.1 * cos(pi * t / 6)]
        #    Here we treat X['CandyA'] as the planned production x_A, and
        #    introduce a nonlinear self-referential upper bound:
        #    X['CandyA'] <= X['CandyA'] * (1 + 0.1 * cos(pi * t / 6))
        #    This is a nonlinear (Type A) modification requested by the problem.
        model.addConstr(Y_t == math.pi * t / 6.0)
        model.addGenConstrCos(Y_t, cos_Y_T)
        seasonal_multiplier = 1.0 + 0.9 * cos_Y_T

        # ❤ Non-linearity is introduced. ❤
        # Originally there was no demand-side constraint on Candy A.
        # We now add a nonlinear self-multiplicative constraint:
        model.addConstr(
            X['CandyA'] <= X['CandyA'] * seasonal_multiplier,
            name=f"SeasonalDemandUpperBound_CandyA_t{t}"
        )

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print(f"Optimal production plan found for month t = {t.X}.")
            print(f"Maximum Total Profit: {model.ObjVal:.2f} Yuan")

            print("\nOptimal Production Quantities (kg):")
            for j in candy_brands:
                if X[j].X > 1e-6:
                    print(f"  {j}: {X[j].X:.2f} kg")
                    print(f"    Composition:")
                    for i in raw_materials:
                        if R[i, j].X > 1e-6:
                            percentage = (R[i, j].X / X[j].X *
                                          100) if X[j].X > 1e-6 else 0
                            print(
                                f"      {i}: {R[i,j].X:.2f} kg ({percentage:.1f}%)"
                            )
                else:
                    print(f"  {j}: 0.00 kg")

            print("\nTotal Raw Material Usage (kg):")
            for i in raw_materials:
                total_used = sum(R[i, j].X for j in candy_brands)
                print(
                    f"  {i}: {total_used:.2f} / {raw_material_limits[i]} kg used"
                )

            print(
                f"\nSeasonal factor for Candy A in month t = {t}: "
                f"{seasonal_multiplier:.4f}"
            )

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. Check constraints and data for contradictions."
            )
        else:
            print(f"Optimization stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # Example: solve for month t = 3 as in the problem statement
    solve_candy_production()
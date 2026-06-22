import gurobipy as gp
from gurobipy import GRB


def solve_candy_production():
    """
    Solves the candy factory production planning problem to maximize profit,
    with an additional non-linear style condition modeled via binary variables:

    If total monthly production of all candies exceeds 4000 kg, then:
      - A one-time fixed extra cost of 5000 Yuan is incurred, and
      - Every kg produced above 4000 kg incurs an additional 0.10 Yuan
        marginal processing cost.

    This is modeled as a mixed-integer linear program (MILP).
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

        # Threshold and extra-cost parameters for the “non-linear” condition
        threshold_total_prod = 4000.0       # kg
        extra_fixed_cost = 5000.0           # Yua  n
        extra_marginal_cost = 0.10          # Yuan/kg (for production above threshold)

        # An upper bound for total production to construct a Big-M
        # (can be chosen as the sum of all raw-material limits, conservative but valid)
        big_M_total_prod = sum(raw_material_limits.values())

        # --- Create Gurobi Model ---
        model = gp.Model("CandyFactoryOptimization_NonlinearCondition")

        # --- Decision Variables ---
        # X[j]: total kilograms of candy brand j produced
        X = model.addVars(
            candy_brands,
            name="X_prod",
            lb=0.0,
            vtype=GRB.CONTINUOUS
        )

        # R[i,j]: kilograms of raw material i used in candy brand j
        R = model.addVars(
            raw_materials,
            candy_brands,
            name="R_raw_in_candy",
            lb=0.0,
            vtype=GRB.CONTINUOUS
        )

        # ❤ Non-linearity is introduced. ❤
        # total_prod = gp.quicksum(X[j] for j in candy_brands)

        # Binary variable indicating whether total production exceeds the threshold
        # y = 1 if total production > threshold_total_prod (in an optimal solution)
        y = model.addVar(vtype=GRB.BINARY, name="y_exceed_threshold")

        # Continuous variable representing the amount of production above the threshold
        # (i.e., max(0, total_prod - threshold_total_prod))
        extra_prod = model.addVar(lb=0.0,
                                  ub=big_M_total_prod,
                                  vtype=GRB.CONTINUOUS,
                                  name="extra_production")

        # Total production variable for convenience
        total_prod = model.addVar(lb=0.0,
                                  ub=big_M_total_prod,
                                  vtype=GRB.CONTINUOUS,
                                  name="total_production")

        # Link total_prod with individual productions
        model.addConstr(
            total_prod == gp.quicksum(X[j] for j in candy_brands),
            name="TotalProductionDef"
        )

        # Logic for extra_prod = max(0, total_prod - threshold_total_prod) using Big-M
        # 1) extra_prod >= total_prod - threshold
        model.addConstr(
            extra_prod >= total_prod - threshold_total_prod,
            name="ExtraProd_LB1"
        )
        # 2) extra_prod <= total_prod - threshold + M*(1 - y)
        model.addConstr(
            extra_prod <= total_prod - threshold_total_prod +
            big_M_total_prod * (1 - y),
            name="ExtraProd_UB1"
        )
        # 3) extra_prod <= M * y
        model.addConstr(
            extra_prod <= big_M_total_prod * y,
            name="ExtraProd_UB2"
        )
        # 4) total_prod - threshold <= M * y
        #    This forces y = 1 if total_prod is strictly above the threshold
        model.addConstr(
            total_prod - threshold_total_prod <= big_M_total_prod * y,
            name="TotalProd_y_link"
        )

        # --- Objective Function: Maximize Total Profit ---
        # Profit = (Sales Revenue - Processing Cost) - Raw Material Cost
        #          - extra_fixed_cost*y - extra_marginal_cost*extra_prod

        # ❤ Non-linearity is introduced. ❤
        # revenue_minus_processing = gp.quicksum(
        #     (selling_prices[j] - processing_fees[j]) * X[j]
        #     for j in candy_brands)
        #
        # total_raw_material_cost = gp.quicksum(raw_material_costs[i] * R[i, j]
        #                                       for i in raw_materials
        #                                       for j in candy_brands)
        #
        # model.setObjective(revenue_minus_processing - total_raw_material_cost,
        #                    GRB.MAXIMIZE)

        # Revenue minus base processing cost
        revenue_minus_processing = gp.quicksum(
            (selling_prices[j] - processing_fees[j]) * X[j]
            for j in candy_brands
        )

        # Raw material cost
        total_raw_material_cost = gp.quicksum(
            raw_material_costs[i] * R[i, j]
            for i in raw_materials
            for j in candy_brands
        )

        # Extra costs when total production exceeds 4000 kg:
        #   - extra_fixed_cost * y
        #   - extra_marginal_cost * extra_prod
        model.setObjective(
            revenue_minus_processing
            - total_raw_material_cost
            - extra_fixed_cost * y
            - extra_marginal_cost * extra_prod,
            GRB.MAXIMIZE
        )

        # --- Constraints ---
        # 1. Mass Balance for Each Candy Brand: Sum of raw materials = Total candy produced
        for j in candy_brands:
            model.addConstr(
                gp.quicksum(R[i, j] for i in raw_materials) == X[j],
                name=f"MassBalance_{j}"
            )

        # 2. Raw Material Availability Limits
        for i in raw_materials:
            model.addConstr(
                gp.quicksum(R[i, j] for j in candy_brands)
                <= raw_material_limits[i],
                name=f"Limit_{i}"
            )

        # 3. Content Percentage Requirements
        # Candy A
        model.addConstr(
            R['RawA', 'CandyA'] >= 0.60 * X['CandyA'],
            name="Content_CandyA_RawA_min"
        )
        model.addConstr(
            R['RawC', 'CandyA'] <= 0.20 * X['CandyA'],
            name="Content_CandyA_RawC_max"
        )

        # Candy B
        model.addConstr(
            R['RawA', 'CandyB'] >= 0.15 * X['CandyB'],
            name="Content_CandyB_RawA_min"
        )
        model.addConstr(
            R['RawC', 'CandyB'] <= 0.60 * X['CandyB'],
            name="Content_CandyB_RawC_max"
        )

        # Candy C
        model.addConstr(
            R['RawC', 'CandyC'] <= 0.50 * X['CandyC'],
            name="Content_CandyC_RawC_max"
        )

        # Suppress Gurobi output if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal production plan found (with threshold-based extra costs).")
            print(f"Maximum Total Profit: {model.ObjVal:.2f} Yuan\n")

            print("Optimal Production Quantities (kg):")
            for j in candy_brands:
                if X[j].X > 1e-6:
                    print(f"  {j}: {X[j].X:.2f} kg")
                    print(f"    Composition:")
                    for i in raw_materials:
                        if R[i, j].X > 1e-6:
                            percentage = (R[i, j].X / X[j].X * 100) if X[j].X > 1e-6 else 0
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

            print("\nAggregate production and extra-cost information:")
            print(f"  Total production: {total_prod.X:.2f} kg")
            print(f"  Threshold (no extra cost up to): {threshold_total_prod:.2f} kg")
            print(f"  Production above threshold: {extra_prod.X:.2f} kg")
            print(f"  Extra capacity activated (y): {int(round(y.X))}")
            if int(round(y.X)) == 1:
                print(f"  Extra fixed cost incurred: {extra_fixed_cost:.2f} Yuan")
                print(
                    f"  Extra marginal cost: {extra_marginal_cost:.2f} Yuan/kg "
                    f"on {extra_prod.X:.2f} kg"
                )
            else:
                print("  No extra fixed or marginal cost incurred (total production ≤ threshold).")

        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Check constraints and data for contradictions.")
            # model.computeIIS()
            # model.write("candy_factory_iis.ilp")
            # print("IIS written to candy_factory_iis.ilp.")
        else:
            print(f"Optimization stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_candy_production()
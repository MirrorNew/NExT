import gurobipy as gp
from gurobipy import GRB


def solve_investment_problem():
    """
    Solves the multi-period investment problem using Gurobi to maximize
    the total principal and interest at the end of the third year.
    This version includes a piecewise (non-linear w.r.t. original text)
    return structure for Project 1, implemented via additional variables
    and constraints so the model remains a MILP.
    """
    try:
        # Initial capital
        K0 = 300000.0

        # Create a new model
        model = gp.Model("MultiPeriodInvestment_PiecewiseP1")

        # =========================
        # --- Decision Variables ---
        # =========================
        # Project 1 (P1): Each year j = 1,2,3 can invest; base return structure (piecewise).
        #
        # x1j is now the TOTAL amount invested in Project 1 in year j:
        x11 = model.addVar(name="x11_P1_Y1", lb=0.0, vtype=GRB.CONTINUOUS)
        x12 = model.addVar(name="x12_P1_Y2", lb=0.0, vtype=GRB.CONTINUOUS)
        x13 = model.addVar(name="x13_P1_Y3", lb=0.0, vtype=GRB.CONTINUOUS)

        # --- Piecewise structure for Project 1 each year ---
        # For each year j, we split x1j into:
        #   y1j: part up to 200,000 (20% return)
        #   y2j: part above 200,000 (10% return)
        # with x1j = y1j + y2j, 0 <= y1j <= 200,000, y2j >= 0

        # Year 1
        y11 = model.addVar(name="y11_P1_Y1_upto200k", lb=0.0, ub=200000.0, vtype=GRB.CONTINUOUS)
        y21 = model.addVar(name="y21_P1_Y1_above200k", lb=0.0, vtype=GRB.CONTINUOUS)

        # Year 2
        y12 = model.addVar(name="y12_P1_Y2_upto200k", lb=0.0, ub=200000.0, vtype=GRB.CONTINUOUS)
        y22 = model.addVar(name="y22_P1_Y2_above200k", lb=0.0, vtype=GRB.CONTINUOUS)

        # Year 3
        y13 = model.addVar(name="y13_P1_Y3_upto200k", lb=0.0, ub=200000.0, vtype=GRB.CONTINUOUS)
        y23 = model.addVar(name="y23_P1_Y3_above200k", lb=0.0, vtype=GRB.CONTINUOUS)

        # Project 2: Start Y1, 2-year, 150% total return (factor 1.5), limit 150k
        x21 = model.addVar(name="x21_P2_Y1", lb=0.0, vtype=GRB.CONTINUOUS)

        # Project 3: Start Y2, 2-year, 160% total return (factor 1.6), limit 200k
        x32 = model.addVar(name="x32_P3_Y2", lb=0.0, vtype=GRB.CONTINUOUS)

        # Project 4: Start Y3, 1-year, 40% profit (return 1.4), limit 100k
        x43 = model.addVar(name="x43_P4_Y3", lb=0.0, vtype=GRB.CONTINUOUS)

        # =========================
        # --- Linking Constraints for P1 piecewise variables ---
        # =========================
        # Ensure x1j is exactly the sum of the two segments y1j and y2j

        model.addConstr(x11 == y11 + y21, name="Link_P1_Y1")
        model.addConstr(x12 == y12 + y22, name="Link_P1_Y2")
        model.addConstr(x13 == y13 + y23, name="Link_P1_Y3")

        # =========================
        # --- Objective Function ---
        # =========================
        # Original (linear, uniform 20% P1):
        # Z = K0 + 0.2*x11 + 0.5*x21 + 0.2*x12 + 0.6*x32 + 0.2*x13 + 0.4*x43
        #
        # ❤ Non-linearity is introduced. ❤
        # New: Project 1 has a piecewise return each year:
        #   - For 0–200,000: 20% return
        #   - For amount above 200,000: 10% return
        #
        # So profit contributions become:
        #   Year 1 P1: 0.2*y11 + 0.1*y21
        #   Year 2 P1: 0.2*y12 + 0.1*y22
        #   Year 3 P1: 0.2*y13 + 0.1*y23

        objective = (
            K0
            + (0.2 * y11 + 0.1 * y21)  # P1 Year 1
            + 0.5 * x21               # P2
            + (0.2 * y12 + 0.1 * y22)  # P1 Year 2
            + 0.6 * x32               # P3
            + (0.2 * y13 + 0.1 * y23)  # P1 Year 3
            + 0.4 * x43               # P4
        )
        model.setObjective(objective, GRB.MAXIMIZE)

        # =========================
        # --- Cash-Flow Constraints ---
        # =========================
        # We must also update the inter-year cash-flow equations because
        # the P1 profit is now piecewise instead of uniform 20%.

        # 1. Cash Availability at Start of Year 1
        # (same – only initial capital split between P1 and P2)
        model.addConstr(x11 + x21 <= K0, "Cash_SOY1")

        # 2. Cash Availability at Start of Year 2
        # Old (uniform 20% on x11): cash_avail_SOY2 = K0 + 0.2*x11 - x21
        # Now: cash_avail_SOY2 = K0 + (0.2*y11 + 0.1*y21) - x21
        # And we require:
        #   x12 + x32 <= cash_avail_SOY2
        # ❤ Non-linearity is introduced. ❤
        model.addConstr(
            x12 + x32
            - (0.2 * y11 + 0.1 * y21) + x21
            <= K0,
            name="Cash_SOY2"
        )

        # 3. Cash Availability at Start of Year 3
        # Old (uniform 20% on x11, x12):
        #   cash_avail_SOY3 = K0 + 0.2*x11 + 0.5*x21 + 0.2*x12 - x32
        #
        # New:
        #   cash_avail_SOY3 = K0
        #                     + (0.2*y11 + 0.1*y21)
        #                     + 0.5*x21
        #                     + (0.2*y12 + 0.1*y22)
        #                     - x32
        #
        # And:
        #   x13 + x43 <= cash_avail_SOY3
        # ❤ Non-linearity is introduced. ❤
        model.addConstr(
            x13 + x43
            - (0.2 * y11 + 0.1 * y21)
            - 0.5 * x21
            - (0.2 * y12 + 0.1 * y22)
            + x32
            <= K0,
            name="Cash_SOY3"
        )

        # 4. Investment Limit for Project 2
        model.addConstr(x21 <= 150000, "Limit_P2")

        # 5. Investment Limit for Project 3
        model.addConstr(x32 <= 200000, "Limit_P3")

        # 6. Investment Limit for Project 4
        model.addConstr(x43 <= 100000, "Limit_P4")

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # =========================
        # --- Results ---
        # =========================
        if model.status == GRB.OPTIMAL:
            print("Optimal investment plan found.")
            print(
                f"Maximum Principal and Interest at End of Year 3: {model.objVal:.2f} yuan"
            )

            total_profit = model.objVal - K0
            print(f"Total Profit over 3 years: {total_profit:.2f} yuan")

            print("\nInvestment Plan Details (yuan):")

            # --- Start of Year 1 ---
            print("  Start of Year 1:")
            print(f"    Project 1 total (P1, piecewise return): {x11.X:.2f}")
            print(f"      Portion @20% (<=200k): {y11.X:.2f}")
            print(f"      Portion @10% (>200k): {y21.X:.2f}")
            print(
                f"    Project 2 (2-year, 150% return, limit 150k): {x21.X:.2f}"
            )

            cash_soy1_invested = x11.X + x21.X
            cash_soy1_uninvested = K0 - cash_soy1_invested
            print(f"    Total invested at SOY1: {cash_soy1_invested:.2f}")
            print(f"    Uninvested cash from SOY1: {cash_soy1_uninvested:.2f}")

            # Cash available at SOY2 based on piecewise return of P1 Y1 and lock-in of P2
            cash_available_soy2 = (
                K0
                + (0.2 * y11.X + 0.1 * y21.X)
                - x21.X
            )
            print(
                f"\n  Cash available at Start of Year 2: {cash_available_soy2:.2f}"
            )

            # --- Start of Year 2 ---
            print("  Start of Year 2:")
            print(f"    Project 1 total (P1, piecewise return): {x12.X:.2f}")
            print(f"      Portion @20% (<=200k): {y12.X:.2f}")
            print(f"      Portion @10% (>200k): {y22.X:.2f}")
            print(
                f"    Project 3 (2-year, 160% return, limit 200k): {x32.X:.2f}"
            )

            cash_soy2_invested = x12.X + x32.X
            cash_soy2_uninvested = cash_available_soy2 - cash_soy2_invested
            print(f"    Total invested at SOY2: {cash_soy2_invested:.2f}")
            print(f"    Uninvested cash from SOY2: {cash_soy2_uninvested:.2f}")

            # Cash available at SOY3 with piecewise P1 Y1 & Y2, plus P2 maturity, minus P3 lock-in
            cash_available_soy3 = (
                K0
                + (0.2 * y11.X + 0.1 * y21.X)
                + 0.5 * x21.X
                + (0.2 * y12.X + 0.1 * y22.X)
                - x32.X
            )
            print(
                f"\n  Cash available at Start of Year 3: {cash_available_soy3:.2f}"
            )

            # --- Start of Year 3 ---
            print("  Start of Year 3:")
            print(f"    Project 1 total (P1, piecewise return): {x13.X:.2f}")
            print(f"      Portion @20% (<=200k): {y13.X:.2f}")
            print(f"      Portion @10% (>200k): {y23.X:.2f}")
            print(
                f"    Project 4 (1-year, 40% profit, limit 100k): {x43.X:.2f}"
            )

            cash_soy3_invested = x13.X + x43.X
            cash_soy3_uninvested = cash_available_soy3 - cash_soy3_invested
            print(f"    Total invested at SOY3: {cash_soy3_invested:.2f}")
            print(
                f"    Uninvested cash from SOY3 (carried to EOY3): {cash_soy3_uninvested:.2f}"
            )

        else:
            print("No optimal solution found. Status code:", model.status)

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
    except AttributeError:
        print(
            "Encountered an attribute error. Gurobi might not be installed or licensed correctly."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_investment_problem()
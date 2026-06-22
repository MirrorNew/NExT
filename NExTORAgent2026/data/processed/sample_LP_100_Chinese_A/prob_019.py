import gurobipy as gp
from gurobipy import GRB


def solve_investment_problem():
    """
    Solves the multi-period investment problem using Gurobi to maximize
    the total principal and interest at the end of the third year.
    This version introduces a nonlinear coupling term for Project (1):
    total principal+interest from Project (1) at end of Year 3 is
    0.00000000001 * x1 * x2 * x3, where x1, x2, x3 are investments in
    Project (1) at the starts of Years 1, 2, and 3, respectively.
    """
    try:
        # Initial capital
        K0 = 300000.0

        # Create a new model (we need a nonlinear model due to the product x1*x2*x3)
        model = gp.Model("MultiPeriodInvestment_Nonlinear")

        # --- Decision Variables ---
        # x_ij: amount invested in project i at the start of year j
        # Project 1: decision variables for three years (x1, x2, x3 in description)
        x11 = model.addVar(name="x11_P1_Y1", lb=0.0,
                           vtype=GRB.CONTINUOUS)  # P1, Start of Year 1  (x1)
        x12 = model.addVar(name="x12_P1_Y2", lb=0.0,
                           vtype=GRB.CONTINUOUS)  # P1, Start of Year 2  (x2)
        x13 = model.addVar(name="x13_P1_Y3", lb=0.0,
                           vtype=GRB.CONTINUOUS)  # P1, Start of Year 3  (x3)

        # Project 2: Start Y1, 2-year, 150% total return (factor 1.5), limit 150k
        x21 = model.addVar(name="x21_P2_Y1", lb=0.0, vtype=GRB.CONTINUOUS)

        # Project 3: Start Y2, 2-year, 160% total return (factor 1.6), limit 200k
        x32 = model.addVar(name="x32_P3_Y2", lb=0.0, vtype=GRB.CONTINUOUS)

        # Project 4: Start Y3, 1-year, 40% profit (return 1.4), limit 100k
        x43 = model.addVar(name="x43_P4_Y3", lb=0.0, vtype=GRB.CONTINUOUS)

        # --- Objective Function ---
        # Original linear objective:
        #   Z = K0 + 0.2*x11 + 0.5*x21 + 0.2*x12 + 0.6*x32 + 0.2*x13 + 0.4*x43
        # Now we replace the "Project (1) contributions" (0.2*x11 + 0.2*x12 + 0.2*x13)
        # with the nonlinear coupled term 0.00000000001 * x11 * x12 * x13.
        #
        # ❤ Non-linearity is introduced. ❤
        # objective = K0 + 0.2 * x11 + 0.5 * x21 + 0.2 * x12 + 0.6 * x32 + 0.2 * x13 + 0.4 * x43

        # New nonlinear objective:
        Y = model.addVar(name="Y", lb=0.0, vtype=GRB.CONTINUOUS)
        model.addConstr(Y == x12 * x13)
        proj1_nonlinear_term = 1e-11 * x11 * Y
        objective = (
            K0
            + proj1_nonlinear_term     # nonlinear principal+interest from Project 1
            + 0.5 * x21                # Project 2 profit part
            + 0.6 * x32                # Project 3 profit part
            + 0.4 * x43                # Project 4 profit part
        )
        model.setObjective(objective, GRB.MAXIMIZE)

        # --- Constraints ---
        # 1. Cash Availability at Start of Year 1
        # All money at the start of year 1 can go into P1 (x11) and P2 (x21)
        model.addConstr(x11 + x21 <= K0, "Cash_SOY1")

        # 2. Cash Availability at Start of Year 2
        # At start of Year 2, available cash:
        #   K0 + 20% return from x11 minus locked capital in P2 (x21)
        #   => available = K0 + 0.2*x11 - x21
        #   investments at SOY2: x12 (P1) + x32 (P3)
        #   so: x12 + x32 <= K0 + 0.2*x11 - x21
        model.addConstr(x12 + x32 - 0.2 * x11 + x21 <= K0, "Cash_SOY2")

        # 3. Cash Availability at Start of Year 3
        # At start of Year 3, available cash:
        #   K0 + 0.2*x11 + 0.5*x21 + 0.2*x12 - x32
        #   (P2 matures with 50% profit at end of Year 2,
        #    P3 is locked in from Year 2 to end of Year 3)
        #   investments at SOY3: x13 (P1) + x43 (P4)
        #   so: x13 + x43 <= K0 + 0.2*x11 + 0.5*x21 + 0.2*x12 - x32
        model.addConstr(
            x13 + x43 - 0.2 * x11 - 0.5 * x21 - 0.2 * x12 + x32 <= K0,
            "Cash_SOY3")

        # 4. Investment Limit for Project 2
        model.addConstr(x21 <= 150000, "Limit_P2")

        # 5. Investment Limit for Project 3
        model.addConstr(x32 <= 200000, "Limit_P3")

        # 6. Investment Limit for Project 4
        model.addConstr(x43 <= 100000, "Limit_P4")

        # Optional: specify that this is a general nonlinear model
        # Gurobi automatically detects it from the nonlinear objective

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal investment plan found (with nonlinear Project 1).")
            print(
                f"Maximum Principal and Interest at End of Year 3: {model.objVal:.2f} yuan"
            )

            total_profit = model.objVal - K0
            print(f"Total Profit over 3 years: {total_profit:.2f} yuan")

            print("\nInvestment Plan Details (yuan):")
            print("  Start of Year 1:")
            print(f"    Project 1 (Nonlinear, coupled over 3 years): {x11.X:.2f}")
            print(
                f"    Project 2 (2-year, 150% return, limit 150k): {x21.X:.2f}"
            )
            cash_soy1_invested = x11.X + x21.X
            cash_soy1_uninvested = K0 - cash_soy1_invested
            print(f"    Total invested at SOY1: {cash_soy1_invested:.2f}")
            print(f"    Uninvested cash from SOY1: {cash_soy1_uninvested:.2f}")

            cash_available_soy2 = K0 + 0.2 * x11.X - x21.X
            print(
                f"\n  Cash available at Start of Year 2: {cash_available_soy2:.2f}"
            )
            print("  Start of Year 2:")
            print(f"    Project 1 (Nonlinear, coupled over 3 years): {x12.X:.2f}")
            print(
                f"    Project 3 (2-year, 160% return, limit 200k): {x32.X:.2f}"
            )
            cash_soy2_invested = x12.X + x32.X
            cash_soy2_uninvested = cash_available_soy2 - cash_soy2_invested
            print(f"    Total invested at SOY2: {cash_soy2_invested:.2f}")
            print(f"    Uninvested cash from SOY2: {cash_soy2_uninvested:.2f}")

            cash_available_soy3 = (
                K0 + 0.2 * x11.X + 0.5 * x21.X + 0.2 * x12.X - x32.X
            )
            print(
                f"\n  Cash available at Start of Year 3: {cash_available_soy3:.2f}"
            )
            print("  Start of Year 3:")
            print(f"    Project 1 (Nonlinear, coupled over 3 years): {x13.X:.2f}")
            print(
                f"    Project 4 (1-year, 40% profit, limit 100k): {x43.X:.2f}"
            )
            cash_soy3_invested = x13.X + x43.X
            cash_soy3_uninvested = cash_available_soy3 - cash_soy3_invested
            print(f"    Total invested at SOY3: {cash_soy3_invested:.2f}")
            print(
                f"    Uninvested cash from SOY3 (carried to EOY3): {cash_soy3_uninvested:.2f}"
            )

            # For clarity, print the nonlinear contribution from Project 1
            proj1_value = 1e-11 * x11.X * x12.X * x13.X
            print(
                f"\n  Nonlinear principal+interest from Project 1 at EOY3: {proj1_value:.6f} yuan"
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
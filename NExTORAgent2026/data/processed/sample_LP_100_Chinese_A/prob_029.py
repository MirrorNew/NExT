import gurobipy as gp
from gurobipy import GRB


def solve_nurse_staffing_with_contract():
    """
    Solves the nurse staffing problem with regular and contract nurses
    when contract nurse wage grows exponentially with the *total*
    number of contract nurses hired (non-linear objective).

    Wage rule:
      - Regular nurse: 10 yuan/hour, fixed
      - Contract nurse: when total number of contract nurses is N,
        each contract nurse is paid 15 * (1.3 ** N) yuan/hour.

    Each nurse (regular or contract) works an 8-hour shift covering 2
    consecutive 4-hour periods. There are 6 possible shift start times.
    """

    try:
        # Create a new model
        model = gp.Model("NurseStaffingContract_Nonlinear")

        # --- Data ---
        # Demands for each 4-hour period (0 to 5)
        # P0: 2:00-6:00, P1: 6:00-10:00, ..., P5: 22:00-2:00
        demands = {
            0: 10,  # 2:00 - 6:00
            1: 15,  # 6:00 - 10:00
            2: 25,  # 10:00 - 14:00
            3: 20,  # 14:00 - 18:00
            4: 18,  # 18:00 - 22:00
            5: 12   # 22:00 - 2:00 (next day)
        }
        num_periods = len(demands)  # Should be 6

        # Shift start times (represented by an index for variables)
        # t=0 starts at 2:00, t=1 at 6:00, ..., t=5 at 22:00
        shift_start_times_desc = [
            "2:00", "6:00", "10:00", "14:00", "18:00", "22:00"
        ]

        # Base costs (per 8-hour shift)
        cost_regular_nurse_shift = 10 * 8  # 10 yuan/hour * 8 hours

        # Base hourly wage for contract nurses before exponential factor
        base_contract_wage_hour = 15
        contract_growth_factor = 1.3

        # --- Decision Variables ---
        # xR[t]: number of regular nurses starting shift at time t
        xR = model.addVars(num_periods, vtype=GRB.INTEGER, name="xR", lb=0)

        # xC[t]: number of contract nurses starting shift at time t
        xC = model.addVars(num_periods, vtype=GRB.INTEGER, name="xC", lb=0)

        # Total number of contract nurses across all start times
        total_contract_nurses = model.addVar(vtype=GRB.INTEGER,
                                             name="TotalContractNurses",
                                             lb=0)

        # Total cost of contract nurses (will be made nonlinear)
        total_contract_cost = model.addVar(vtype=GRB.CONTINUOUS,
                                           name="TotalContractCost",
                                           lb=0)

        # --- Linking constraints for aggregation ---
        # total_contract_nurses = sum_t xC[t]
        model.addConstr(
            total_contract_nurses ==
            gp.quicksum(xC[t] for t in range(num_periods)),
            name="TotalContractNursesDef"
        )

        # --- Objective Function ---

        # Regular nurse cost is linear
        total_regular_cost = gp.quicksum(
            cost_regular_nurse_shift * xR[t] for t in range(num_periods)
        )

        # ❤ Non-linearity is introduced. ❤
        # The contract nurse wage per hour is:
        #   base_contract_wage_hour * (contract_growth_factor ** total_contract_nurses)
        # Each contract nurse works 8 hours, and there are total_contract_nurses of them.
        # So total contract cost:
        #   total_contract_cost = total_contract_nurses
        #                         * base_contract_wage_hour
        #                         * (contract_growth_factor ** total_contract_nurses)
        #
        # We encode this via a general constraint (nonlinear) so that
        # Gurobi treats the model as a general non-linear program.
        model.addGenConstrExp(
            total_contract_nurses * gp.log(contract_growth_factor),
            total_contract_cost,
            name="ExpContractCost_Internal"
        )

        # Above constraint defines: total_contract_cost = exp( total_contract_nurses * log(1.3) )
        # But we need:
        #   total_contract_cost = total_contract_nurses
        #                         * base_contract_wage_hour
        #                         * (contract_growth_factor ** total_contract_nurses)
        # So we introduce an additional continuous variable to hold 1.3^N,
        # then scale by 15 * 8 * N.

        contract_exp_term = model.addVar(vtype=GRB.CONTINUOUS,
                                         name="ContractGrowthTerm",
                                         lb=0)

        # Exp for 1.3^N
        # ❤ Non-linearity is introduced. ❤
        model.addGenConstrExp(
            total_contract_nurses * gp.log(contract_growth_factor),
            contract_exp_term,
            name="ContractExpTermDef"
        )

        # Now re-define total_contract_cost properly:
        # ❤ Non-linearity is introduced. ❤
        # Comment out the previous generic use of total_contract_cost in objective
        # and instead enforce:
        #   total_contract_cost = total_contract_nurses
        #                         * base_contract_wage_hour * 8
        #                         * contract_exp_term
        model.addConstr(
            total_contract_cost ==
            total_contract_nurses * base_contract_wage_hour * 8 * contract_exp_term,
            name="TotalContractCostDef"
        )

        # Final objective: minimize total cost (regular + contract)
        # ❤ Non-linearity is introduced. ❤
        # model.setObjective(
        #     gp.quicksum(cost_regular_nurse_shift * xR[t] +
        #                 cost_contract_nurse_shift * xC[t]
        #                 for t in range(num_periods)),
        #     GRB.MINIMIZE
        # )
        model.setObjective(
            total_regular_cost + total_contract_cost,
            GRB.MINIMIZE
        )

        # --- Constraints ---
        # Demand coverage for each period.
        # Each nurse works for 8 hours, covering two 4-hour periods.

        # Period 0 (2:00-6:00): Covered by staff starting at 22:00 (t=5) and 2:00 (t=0)
        model.addConstr(
            (xR[5] + xC[5]) + (xR[0] + xC[0]) >= demands[0],
            "Demand_P0"
        )

        # Period 1 (6:00-10:00): Covered by staff starting at 2:00 (t=0) and 6:00 (t=1)
        model.addConstr(
            (xR[0] + xC[0]) + (xR[1] + xC[1]) >= demands[1],
            "Demand_P1"
        )

        # Period 2 (10:00-14:00): Covered by staff starting at 6:00 (t=1) and 10:00 (t=2)
        model.addConstr(
            (xR[1] + xC[1]) + (xR[2] + xC[2]) >= demands[2],
            "Demand_P2"
        )

        # Period 3 (14:00-18:00): Covered by staff starting at 10:00 (t=2) and 14:00 (t=3)
        model.addConstr(
            (xR[2] + xC[2]) + (xR[3] + xC[3]) >= demands[3],
            "Demand_P3"
        )

        # Period 4 (18:00-22:00): Covered by staff starting at 14:00 (t=3) and 18:00 (t=4)
        model.addConstr(
            (xR[3] + xC[3]) + (xR[4] + xC[4]) >= demands[4],
            "Demand_P4"
        )

        # Period 5 (22:00-2:00): Covered by staff starting at 18:00 (t=4) and 22:00 (t=5)
        model.addConstr(
            (xR[4] + xC[4]) + (xR[5] + xC[5]) >= demands[5],
            "Demand_P5"
        )

        # Optional: limit total_contract_nurses to a reasonable bound
        # to help the nonlinear solver. For example, cannot exceed
        # total demand over the day.
        max_possible_nurses = sum(demands.values())
        model.addConstr(
            total_contract_nurses <= max_possible_nurses,
            name="UpperBoundTotalContract"
        )

        # Suppress Gurobi output to console (optional)
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal staffing plan found (with nonlinear contract wage).")
            print(f"Minimum Total Daily Wage Cost: {model.objVal:.2f} Yuan")

            total_regular_nurses_value = sum(xR[t].X for t in range(num_periods))
            total_contract_nurses_value = total_contract_nurses.X

            print(f"\nTotal Regular Nurses to Hire: {total_regular_nurses_value:.0f}")
            print(f"Total Contract Nurses to Hire: {total_contract_nurses_value:.0f}")

            if total_contract_nurses_value > 0.5:
                # use tolerance to avoid numerical issues
                print("\nDecision: The hospital SHOULD hire contract nurses.")
            else:
                print(
                    "\nDecision: The hospital does NOT need to hire contract nurses based on cost minimization."
                )

            print("\nNumber of Nurses Starting at Each Shift:")
            print(f"{'Start Time':<12} | {'Regular':<10} | {'Contract':<10}")
            print("-" * 40)
            for t in range(num_periods):
                print(
                    f"{shift_start_times_desc[t]:<12} | {xR[t].X:<10.0f} | {xC[t].X:<10.0f}"
                )

            print("\nVerification of Coverage per Period:")
            coverage = [0] * num_periods
            coverage[0] = (xR[5].X + xC[5].X) + (xR[0].X + xC[0].X)
            coverage[1] = (xR[0].X + xC[0].X) + (xR[1].X + xC[1].X)
            coverage[2] = (xR[1].X + xC[1].X) + (xR[2].X + xC[2].X)
            coverage[3] = (xR[2].X + xC[2].X) + (xR[3].X + xC[3].X)
            coverage[4] = (xR[3].X + xC[3].X) + (xR[4].X + xC[4].X)
            coverage[5] = (xR[4].X + xC[4].X) + (xR[5].X + xC[5].X)

            period_desc = [
                "2:00-6:00", "6:00-10:00", "10:00-14:00", "14:00-18:00",
                "18:00-22:00", "22:00-2:00"
            ]
            for p in range(num_periods):
                print(
                    f"  Period {period_desc[p]} (Demand: {demands[p]}): "
                    f"Covered by {coverage[p]:.0f} nurses"
                )

            print(f"\nTotal contract wage (nonlinear): {total_contract_cost.X:.2f} Yuan")
            print(f"Total contract nurses (N): {total_contract_nurses.X:.0f}")
            print(f"Growth term 1.3^N: {contract_exp_term.X:.4f}")

        else:
            print("No optimal solution found. Status code:", model.status)

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
    except AttributeError:
        print(
            "Encountered an attribute error, Gurobi might not be installed or licensed correctly."
        )


if __name__ == '__main__':
    solve_nurse_staffing_with_contract()
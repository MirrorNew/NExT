import gurobipy as gp
from gurobipy import GRB


def solve_forklift_leasing(
    demand=[6, 9, 12, 8, 5, 7, 4],
    long_term_cost=240,
    short_term_cost=390,
    shared_cost=220,
    shared_weeks=[2, 4, 5],
    promo_cost=190,
    promo_max_num=2,
    promo_weeks=[4, 5, 6, 7],
    # new parameter for the fixed dispatching cost
    dispatch_fixed_cost=800,
    short_term_threshold=5
):
    """
    Models and solves the forklift leasing optimization problem.
    """
    # --- 1. Model Creation ---
    model = gp.Model("ForkliftLeasing")

    # --- 2. Parameters & Sets ---
    week_num = len(demand)
    weeks = range(1, week_num + 1)
    long_term_weeks = week_num

    # --- 3. Decision Variables ---
    long_term_num = model.addVar(vtype=GRB.INTEGER, name="LongTermNum")
    short_term_num = model.addVars(weeks, vtype=GRB.INTEGER, name="ShortTermNum")
    shared_num = model.addVar(vtype=GRB.INTEGER, name="SharedNum")
    promo_num = model.addVars(weeks, vtype=GRB.INTEGER, name="PromoNum")

    # Binary variable: whether week i triggers the extra 800 Yuan fixed cost
    dispatch_used = model.addVars(weeks, vtype=GRB.BINARY, name="DispatchUsed")

    # --- 4. Objective Function ---
    objective = long_term_cost * long_term_num * long_term_weeks
    objective += short_term_cost * gp.quicksum(short_term_num[i] for i in weeks)
    objective += shared_cost * shared_num * len(shared_weeks)
    objective += promo_cost * gp.quicksum(
        promo_num[i] for i in weeks if i in promo_weeks
    )

    # ❤ Non-linearity is introduced. ❤
    # Original objective part (without fixed dispatching cost) is commented out
    # objective += 0  # placeholder for original linear objective (no fixed cost)

    # Add weekly fixed dispatching management cost (potential source of non-linearity)
    # If in week i, short_term_num[i] > 5, then dispatch_used[i] should be 1,
    # and we pay an extra fixed cost of 800 that week.
    objective += dispatch_fixed_cost * gp.quicksum(dispatch_used[i] for i in weeks)

    model.setObjective(objective, GRB.MINIMIZE)

    # --- 5. Constraints ---
    # Constraint 1: Total forklifts per week must meet demand (non-shared weeks)
    for i in weeks:
        if i not in shared_weeks:
            model.addConstr(
                long_term_num + short_term_num[i] + promo_num[i] >= demand[i - 1],
                name=f"Demand_Week_{i}",
            )

    # Constraint 2: Total forklifts during shared weeks must meet demand
    for i in shared_weeks:
        model.addConstr(
            long_term_num + short_term_num[i] + shared_num + promo_num[i]
            >= demand[i - 1],
            name=f"Demand_SharedWeek_{i}",
        )

    # Constraint 3: Promotional forklift quantity must not exceed maximum limit
    for i in promo_weeks:
        model.addConstr(
            promo_num[i] <= promo_max_num, name=f"PromoLimit_Week_{i}"
        )

    # Constraint 4: Non-promotional weeks cannot lease promotional forklifts
    for i in weeks:
        if i not in promo_weeks:
            model.addConstr(promo_num[i] == 0, name=f"NoPromo_Week_{i}")

    # --- 6. Non-linear behavior for short-term forklifts (via binary linking) ---
    # We introduce a big-M style logic to approximate:
    # dispatch_used[i] = 1  if short_term_num[i] >= 6
    # dispatch_used[i] = 0  if short_term_num[i] <= 5
    #
    # Short-term forklifts are integers, dispatch_used is binary, which makes
    # the model a mixed-integer program; the cost term dispatch_fixed_cost *
    # dispatch_used[i] creates the stepwise (non-smooth) cost effect.

    M = 1000  # sufficiently large upper bound on short_term_num per week

    for i in weeks:
        # If dispatch_used[i] = 0, then short_term_num[i] <= short_term_threshold
        model.addConstr(
            short_term_num[i] <= short_term_threshold + M * dispatch_used[i],
            name=f"DispatchUpperLink_Week_{i}",
        )

        # If short_term_num[i] >= short_term_threshold + 1, then dispatch_used[i] must be 1.
        # Implement: short_term_num[i] >= short_term_threshold + 1 - M * (1 - dispatch_used[i])
        model.addConstr(
            short_term_num[i]
            >= (short_term_threshold + 1) - M * (1 - dispatch_used[i]),
            name=f"DispatchLowerLink_Week_{i}",
        )

    # --- 7. Solve the Model ---
    model.optimize()

    # --- 8. Return Results ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "long_term_num": long_term_num.X,
            "shared_num": shared_num.X,
            "short_term_num": {i: short_term_num[i].X for i in weeks},
            "promo_num": {i: promo_num[i].X for i in weeks},
            "dispatch_used": {i: dispatch_used[i].X for i in weeks},
        }
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_forklift_leasing()
    print(result)
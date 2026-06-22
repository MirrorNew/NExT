import gurobipy as gp
from gurobipy import GRB


def solve_forklift_leasing(
    ):
    demand = [6, 9, 12, 8, 5, 7, 4]
    long_term_cost = 240  # kept as parameter but no longer used directly in the objective
    short_term_cost = 390
    shared_cost = 220
    shared_weeks = [2, 4, 5]
    promo_cost = 190
    promo_max_num = 2
    promo_weeks = [4, 5, 6, 7]
    """
    Models and solves the forklift leasing optimization problem
    with a non-linear long-term leasing cost:
        Long-term total cost = 10 * Π_{i=1}^7 L_i
    where L_i is the number of long-term forklifts in week i.
    """
    # --- 1. Model Creation ---
    model = gp.Model("ForkliftLeasing_Nonlinear")

    # --- 2. Parameters & Sets ---
    week_num = len(demand)
    weeks = range(1, week_num + 1)

    # --- 3. Decision Variables ---

    # ❤ Non-linearity is introduced. ❤
    # long_term_num = model.addVar(vtype=GRB.INTEGER, name="LongTermNum")
    # Replace the single long-term variable with week-dependent long-term variables L_i
    long_term_num = model.addVars(week_num, vtype=GRB.INTEGER, name="LongTermNum")

    short_term_num = model.addVars(weeks, vtype=GRB.INTEGER, name="ShortTermNum")
    shared_num = model.addVar(vtype=GRB.INTEGER, name="SharedNum")
    promo_num = model.addVars(weeks, vtype=GRB.INTEGER, name="PromoNum")

    # --- 4. Objective Function ---

    # ❤ Non-linearity is introduced. ❤
    # Original linear long-term cost:
    # long_term_weeks = week_num
    # objective = long_term_cost * long_term_num * long_term_weeks
    # New non-linear long-term cost: 10 * Π_{i=1}^7 L_i
    # (We build the product over all weeks)
    x12 = model.addVar(lb=1)
    x34 = model.addVar(lb=1)
    x56 = model.addVar(lb=1)
    x1234 = model.addVar(lb=1)
    x567 = model.addVar(lb=1)
    model.addConstr(x12 == long_term_num[0] * long_term_num[1])
    model.addConstr(x34 == long_term_num[2] * long_term_num[3])
    model.addConstr(x56 == long_term_num[4] * long_term_num[5])
    model.addConstr(x1234 == x12 * x34)
    model.addConstr(x567 == x56 * long_term_num[6])

    objective = 5 * x1234 * x567

    # Add linear short-term cost
    objective += short_term_cost * gp.quicksum(short_term_num[i] for i in weeks)

    # Shared forklifts cost (only available in shared_weeks)
    objective += shared_cost * shared_num * len(shared_weeks)

    # Promotional forklifts cost
    objective += promo_cost * gp.quicksum(promo_num[i] for i in weeks if i in promo_weeks)

    model.setObjective(objective, GRB.MINIMIZE)

    # --- 5. Constraints ---
    # Constraint 1: Total forklifts per week must meet demand (non-shared weeks)
    for i in weeks:
        if i not in shared_weeks:
            # ❤ Non-linearity is introduced. ❤
            # model.addConstr(long_term_num + short_term_num[i] + promo_num[i]
            #                 >= demand[i - 1],
            #                 name=f"Demand_Week_{i}")
            # Replace long_term_num with week-specific long_term_num[i]
            model.addConstr(long_term_num[i-1] + short_term_num[i] + promo_num[i]
                            >= demand[i - 1],
                            name=f"Demand_Week_{i}")

    # Constraint 2: Total forklifts during shared weeks must meet demand
    for i in shared_weeks:
        # ❤ Non-linearity is introduced. ❤
        # model.addConstr(long_term_num + short_term_num[i] + shared_num +
        #                 promo_num[i] >= demand[i - 1],
        #                 name=f"Demand_SharedWeek_{i}")
        # Replace long_term_num with week-specific long_term_num[i]
        model.addConstr(long_term_num[i-1] + short_term_num[i] + shared_num +
                        promo_num[i] >= demand[i - 1],
                        name=f"Demand_SharedWeek_{i}")

    # Constraint 3: Promotional forklift quantity must not exceed maximum limit
    for i in promo_weeks:
        model.addConstr(promo_num[i] <= promo_max_num, name=f"PromoLimit_Week_{i}")

    # Constraint 4: Non-promotional weeks cannot lease promotional forklifts
    for i in weeks:
        if i not in promo_weeks:
            model.addConstr(promo_num[i] == 0, name=f"NoPromo_Week_{i}")

    # Note: Long-term variables L_i are implicitly constrained to be >= 0
    # because they are integer variables (default lower bound 0).

    # --- 6. Solve the Model ---
    # This is a non-convex MINLP due to the product term; enable non-convex handling
    model.Params.NonConvex = 2
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        solution = {
            "status": "optimal",
            "obj": model.ObjVal,
            "long_term": {i: int(long_term_num[i-1].X) for i in weeks},
            "short_term": {i: int(short_term_num[i].X) for i in weeks},
            "shared": int(shared_num.X),
            "promo": {i: int(promo_num[i].X) for i in weeks},
        }
        return solution
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_forklift_leasing()
    print(result)
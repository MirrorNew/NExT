import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Solve the MILP insurance planning problem with Gurobi
# ============================================================

# 1. Define all parameter matrices and data inputs
def build_parameters():
    # ---- Given parameter list ----
    num_factories = 3
    factories = ['A', 'B', 'C']
    num_plans = 3
    plans = ['A', 'B', 'C']
    startup_cost = {'A': 50, 'B': 55, 'C': 40}
    num_cities = 10
    approx_customers_per_city = 600
    avg_deductible_max = 1100.0
    transportation_cost_per_unit = 0
    min_total_premium_income = 1000000
    age_groups = ['0-6', '6-12', '12-18', '18-35', '35-65', '65-100']
    min_share_18_35 = {'A': 0.25, 'B': 0.3, 'C': 0.2}
    min_fraction_city_population_per_subquota = 0.0166666667
    city_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    city_population = {
        'A': 600, 'B': 650, 'C': 620, 'D': 600, 'E': 560,
        'F': 540, 'G': 590, 'H': 580, 'I': 610, 'J': 500
    }
    Table_1_City_Population = {
        'City': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'Number_of_People': [600, 650, 620, 600, 560, 540, 590, 580, 610, 500]
    }
    Table_2_Premium_Age_Data = [
        {'Plan': 'A', 'Age_Group_Years': '0-6', 'Premium_USD': 200, 'Deductible_USD': 900,
         'Doctor_Visit_Fees_USD': 30, 'Specialist_Fees_USD': 70},
        {'Plan': 'A', 'Age_Group_Years': '6-12', 'Premium_USD': 200, 'Deductible_USD': 950,
         'Doctor_Visit_Fees_USD': 28, 'Specialist_Fees_USD': 68},
        {'Plan': 'A', 'Age_Group_Years': '12-18', 'Premium_USD': 200, 'Deductible_USD': 1000,
         'Doctor_Visit_Fees_USD': 28, 'Specialist_Fees_USD': 70},
        # Missing plan label for A, 18-35
        {'Plan': None, 'Age_Group_Years': None, 'Premium_USD': 200, 'Deductible_USD': 1000,
         'Doctor_Visit_Fees_USD': 35, 'Specialist_Fees_USD': 70},
        {'Plan': 'A', 'Age_Group_Years': '65-100', 'Premium_USD': 200, 'Deductible_USD': 1100,
         'Doctor_Visit_Fees_USD': 35, 'Specialist_Fees_USD': 75},
        {'Plan': 'B', 'Age_Group_Years': '0-6', 'Premium_USD': 180, 'Deductible_USD': 1150,
         'Doctor_Visit_Fees_USD': 38, 'Specialist_Fees_USD': 80},
        {'Plan': 'B', 'Age_Group_Years': '6-12', 'Premium_USD': 180, 'Deductible_USD': 1180,
         'Doctor_Visit_Fees_USD': 39, 'Specialist_Fees_USD': 75},
        {'Plan': 'B', 'Age_Group_Years': '12-18', 'Premium_USD': 180, 'Deductible_USD': 1200,
         'Doctor_Visit_Fees_USD': 40, 'Specialist_Fees_USD': 75},
        {'Plan': 'B', 'Age_Group_Years': '18-35', 'Premium_USD': 180, 'Deductible_USD': 1200,
         'Doctor_Visit_Fees_USD': 40, 'Specialist_Fees_USD': 80},
        {'Plan': 'B', 'Age_Group_Years': '35-65', 'Premium_USD': 180, 'Deductible_USD': 1200,
         'Doctor_Visit_Fees_USD': 40, 'Specialist_Fees_USD': 85},
        # Missing plan label for C, 0-6
        {'Plan': None, 'Age_Group_Years': None, 'Premium_USD': 220, 'Deductible_USD': 750,
         'Doctor_Visit_Fees_USD': 25, 'Specialist_Fees_USD': 65},
        {'Plan': 'C', 'Age_Group_Years': '6-12', 'Premium_USD': 220, 'Deductible_USD': 780,
         'Doctor_Visit_Fees_USD': 22, 'Specialist_Fees_USD': 63},
        {'Plan': 'C', 'Age_Group_Years': '12-18', 'Premium_USD': 220, 'Deductible_USD': 800,
         'Doctor_Visit_Fees_USD': 20, 'Specialist_Fees_USD': 60},
        # Missing plan label for C, 35-65
        {'Plan': None, 'Age_Group_Years': '35-65', 'Premium_USD': 220, 'Deductible_USD': 850,
         'Doctor_Visit_Fees_USD': 22, 'Specialist_Fees_USD': 63},
        {'Plan': 'C', 'Age_Group_Years': '65-100', 'Premium_USD': 220, 'Deductible_USD': 875,
         'Doctor_Visit_Fees_USD': 25, 'Specialist_Fees_USD': 65}
    ]
    premium_values = [200, 180, 220]
    deductible_values = [900, 950, 1000, 1100, 1150, 1180, 1200, 750, 780, 800, 850, 875]
    doctor_visit_fee_values = [30, 28, 35, 38, 39, 40, 25, 22, 20]
    specialist_fee_values = [70, 68, 75, 80, 85, 65, 63, 60]

    # ---- Build π_{p,a}, d_{p,a}, v_{p,a}, s_{p,a} from the table ----
    premium = {(p, a): 0.0 for p in plans for a in age_groups}
    deduct = {(p, a): 0.0 for p in plans for a in age_groups}
    visit = {(p, a): 0.0 for p in plans for a in age_groups}
    spec = {(p, a): 0.0 for p in plans for a in age_groups}

    def set_row(plan_, age_, prem_, ded_, vis_, sp_):
        premium[(plan_, age_)] = prem_
        deduct[(plan_, age_)] = ded_
        visit[(plan_, age_)] = vis_
        spec[(plan_, age_)] = sp_

    # Direct entries
    for row in Table_2_Premium_Age_Data:
        plan = row['Plan']
        age = row['Age_Group_Years']
        if plan is not None and age is not None:
            set_row(plan, age,
                    row['Premium_USD'],
                    row['Deductible_USD'],
                    row['Doctor_Visit_Fees_USD'],
                    row['Specialist_Fees_USD'])

    # Missing (A, 18-35)
    for row in Table_2_Premium_Age_Data:
        if row['Plan'] is None and row['Age_Group_Years'] is None and row['Premium_USD'] == 200:
            set_row('A', '18-35',
                    row['Premium_USD'],
                    row['Deductible_USD'],
                    row['Doctor_Visit_Fees_USD'],
                    row['Specialist_Fees_USD'])

    # Missing (C, 0-6)
    for row in Table_2_Premium_Age_Data:
        if row['Plan'] is None and row['Age_Group_Years'] is None and row['Premium_USD'] == 220:
            set_row('C', '0-6',
                    row['Premium_USD'],
                    row['Deductible_USD'],
                    row['Doctor_Visit_Fees_USD'],
                    row['Specialist_Fees_USD'])

    # Missing (C, 35-65)
    for row in Table_2_Premium_Age_Data:
        if row['Plan'] is None and row['Age_Group_Years'] == '35-65' and row['Premium_USD'] == 220:
            set_row('C', '35-65',
                    row['Premium_USD'],
                    row['Deductible_USD'],
                    row['Doctor_Visit_Fees_USD'],
                    row['Specialist_Fees_USD'])

    # Derived parameters
    N_tot = sum(city_population[c] for c in city_names)
    # L_c = floor(N_c / 60)
    L = {c: int(city_population[c] // 60) for c in city_names}
    # Big-M, use N_tot
    M = {p: N_tot for p in plans}

    data = {
        "num_factories": num_factories,
        "factories": factories,
        "num_plans": num_plans,
        "plans": plans,
        "startup_cost": startup_cost,
        "num_cities": num_cities,
        "approx_customers_per_city": approx_customers_per_city,
        "avg_deductible_max": avg_deductible_max,
        "transportation_cost_per_unit": transportation_cost_per_unit,
        "min_total_premium_income": min_total_premium_income,
        "age_groups": age_groups,
        "min_share_18_35": min_share_18_35,
        "min_fraction_city_population_per_subquota": min_fraction_city_population_per_subquota,
        "city_names": city_names,
        "city_population": city_population,
        "Table_1_City_Population": Table_1_City_Population,
        "Table_2_Premium_Age_Data": Table_2_Premium_Age_Data,
        "premium_values": premium_values,
        "deductible_values": deductible_values,
        "doctor_visit_fee_values": doctor_visit_fee_values,
        "specialist_fee_values": specialist_fee_values,
        "premium": premium,
        "deduct": deduct,
        "visit": visit,
        "spec": spec,
        "N_tot": N_tot,
        "L": L,
        "M": M
    }
    return data


def main():
    params = build_parameters()
    plans = params["plans"]
    city_names = params["city_names"]
    age_groups = params["age_groups"]
    startup_cost = params["startup_cost"]
    city_population = params["city_population"]
    avg_deductible_max = params["avg_deductible_max"]
    min_total_premium_income = params["min_total_premium_income"]
    min_share_18_35 = params["min_share_18_35"]
    premium = params["premium"]
    deduct = params["deduct"]
    visit = params["visit"]
    spec = params["spec"]
    N_tot = params["N_tot"]
    L = params["L"]
    M = params["M"]

    # 2. Create Gurobi model
    model = gp.Model("Insurance_Plan_Allocation")

    # 3. Create decision variables
    # x_{c,a,p} : integer >= 0
    x = model.addVars(
        city_names,
        age_groups,
        plans,
        vtype=GRB.INTEGER,
        name="x"
    )

    # X_p: total customers per plan
    X = model.addVars(
        plans,
        vtype=GRB.INTEGER,
        name="X"
    )

    # y_p: startup indicator
    y = model.addVars(
        plans,
        vtype=GRB.BINARY,
        name="y"
    )

    # 4. Set up the objective function
    # Minimize sum x * (d+v+s) + startup costs
    cost_expr = gp.LinExpr()
    for c in city_names:
        for a in age_groups:
            for p in plans:
                per_cost = deduct[(p, a)] + visit[(p, a)] + spec[(p, a)]
                cost_expr += per_cost * x[c, a, p]
    for p in plans:
        cost_expr += startup_cost[p] * y[p]

    model.setObjective(cost_expr, GRB.MINIMIZE)

    # 5. Add all constraints

    # (1) Demand satisfaction by city
    for c in city_names:
        model.addConstr(
            gp.quicksum(x[c, a, p] for a in age_groups for p in plans) == city_population[c],
            name=f"demand_city_{c}"
        )

    # (2) Global demand total
    model.addConstr(
        gp.quicksum(x[c, a, p] for c in city_names for a in age_groups for p in plans) == N_tot,
        name="total_customers"
    )

    # (3) Minimum sub-quota per (city, age, plan)
    for c in city_names:
        for a in age_groups:
            for p in plans:
                model.addConstr(
                    x[c, a, p] >= L[c],
                    name=f"min_subquota_{c}_{a}_{p}"
                )

    # (4) Plan-wise total customers
    for p in plans:
        model.addConstr(
            X[p] == gp.quicksum(x[c, a, p] for c in city_names for a in age_groups),
            name=f"plan_total_{p}"
        )

    # (5) Average deductible constraint
    model.addConstr(
        gp.quicksum(
            x[c, a, p] * deduct[(p, a)]
            for c in city_names for a in age_groups for p in plans
        ) <= avg_deductible_max * N_tot,
        name="avg_deductible"
    )

    # (6) Revenue (premium) constraint
    model.addConstr(
        gp.quicksum(
            x[c, a, p] * premium[(p, a)]
            for c in city_names for a in age_groups for p in plans
        ) >= min_total_premium_income,
        name="premium_revenue"
    )

    # (7) Young market penetration - Plan A
    model.addConstr(
        gp.quicksum(x[c, '18-35', 'A'] for c in city_names) >=
        min_share_18_35['A'] * X['A'],
        name="young_penetration_A"
    )

    # (8) Young market penetration - Plan B
    model.addConstr(
        gp.quicksum(x[c, '18-35', 'B'] for c in city_names) >=
        min_share_18_35['B'] * X['B'],
        name="young_penetration_B"
    )

    # (9) Young market penetration - Plan C
    model.addConstr(
        gp.quicksum(x[c, '18-35', 'C'] for c in city_names) >=
        min_share_18_35['C'] * X['C'],
        name="young_penetration_C"
    )

    # (10) Startup logic indicator constraints
    for p in plans:
        # y_p = 0 => X_p = 0
        model.addGenConstrIndicator(y[p], 0, X[p] == 0, name=f"startup_off_{p}")
        # y_p = 1 => X_p <= M_p
        model.addGenConstrIndicator(y[p], 1, X[p] <= M[p], name=f"startup_on_{p}")

    # 6. Solve the model
    model.optimize()

    # 7. Print results and final answer (total cost)
    if model.Status == GRB.OPTIMAL:
        total_cost = model.ObjVal
        print(f"Optimal objective (Total Cost): {total_cost:.2f}")
        print("Startup decisions (y_p) and total customers X_p:")
        for p in plans:
            print(f"  Plan {p}: y = {int(round(y[p].X))}, X = {int(round(X[p].X))}")
        print("\nNon-zero assignments x_{c,a,p}:")
        for c in city_names:
            for a in age_groups:
                for p in plans:
                    val = x[c, a, p].X
                    if val > 1e-6:
                        print(f"  City {c}, Age {a}, Plan {p}: {int(round(val))} customers")
        # FinalAnswer is the total costs (objective value)
        print(f"FinalAnswer=【{total_cost:.2f}】")
    else:
        print(f"Model ended with status {model.Status}, no optimal solution reported.")
        # Still print something for FinalAnswer to comply with format
        print("FinalAnswer=【NaN】")


if __name__ == "__main__":
    main()
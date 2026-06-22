import gurobipy as gp

# Complete Gurobi model to solve the Rainbow Group goal programming problem
# and answer: how many people in category 1 are assigned to Production in City A (x_{1,1,1})?

# All numeric / structural values are taken from the provided Parameters List.


def solve_rainbow_group_goal_programming():
    # =========================
    # 1. Basic parameters
    # =========================
    company_name = "Rainbow Group"
    establishment_year = 1995
    business_areas = ["home appliances", "new energy equipment", "supporting services"]
    cities = ["A", "B"]
    facility_types = ["production bases", "marketing centers"]
    planning_horizon_name = "14th Five-Year Plan"
    core_positions = ["production", "marketing", "finance"]
    number_of_core_positions = 3
    headquarters_city = "A"
    new_smart_factory_city = "B"
    total_candidates_screened = 180
    category_3_size = 30
    category_3_suitable_majors = ["production", "finance"]
    category_3_preferred_major = "production"
    city_B_actual_demand_major = "finance"
    total_recruits_planned = 170
    number_of_professions = 3
    recruitment_cities = ["A", "B"]
    qualified_applicants_total = 180
    number_of_categories = 6
    categorization_criteria = ["suitable major", "desired major", "desired city"]
    p1_description = "recruit employees who are suitable for the job and profession"
    p2_description = "≥80% recruited in desired profession"
    p2_percentage_threshold = 0.8
    p3_description = "≥80% recruited in desired city"
    p3_percentage_threshold = 0.8
    priority_relation = "p1 >> p2 >> p3"
    p1_weight = 10000
    p2_weight = 100
    p3_weight = 1
    i_range = 6
    j_major_mapping = {1: "production", 2: "marketing", 3: "finance"}
    k_city_mapping = {1: "A", 2: "B"}
    deviation_index_range_start = 1
    deviation_index_range_end = 7
    number_of_deviation_pairs = 7

    # Table C-24 recruitment plan (not used programmatically except for comments/consistency)
    Table_1_C24_recruitment_plan = [
        {"city": "A", "profession": "Production", "number_of_recruits": 20},
        {"city": "A", "profession": "Marketing", "number_of_recruits": 30},
        {"city": "A", "profession": "Finance", "number_of_recruits": 40},
        {"city": "B", "profession": "Production", "number_of_recruits": 25},
        {"city": "B", "profession": "Marketing", "number_of_recruits": 20},
        {"city": "B", "profession": "Finance", "number_of_recruits": 35},
    ]
    recruitment_cities_C24 = ["A", "B"]
    recruitment_professions_C24 = ["Production", "Marketing", "Finance"]
    recruits_by_city_and_profession = {
        "A_Production": 20,
        "A_Marketing": 30,
        "A_Finance": 40,
        "B_Production": 25,
        "B_Marketing": 20,
        "B_Finance": 35,
    }

    # Table C-25 applicant categories (for reference; constraints encoded directly)
    Table_2_C25_applicant_categories = [
        {
            "category": 1,
            "number_of_people": 30,
            "suitable_majors": ["Production", "Marketing"],
            "desired_major": "Production",
            "desired_city": "A",
        },
        {
            "category": 2,
            "number_of_people": 30,
            "suitable_majors": ["Marketing", "Finance"],
            "desired_major": "Marketing",
            "desired_city": "A",
        },
        {
            "category": 3,
            "number_of_people": 30,
            "suitable_majors": ["Production", "Finance"],
            "desired_major": "Production",
            "desired_city": "B",
        },
        {
            "category": 4,
            "number_of_people": 30,
            "suitable_majors": ["Production", "Finance"],
            "desired_major": "Finance",
            "desired_city": "B",
        },
        {
            "category": 5,
            "number_of_people": 30,
            "suitable_majors": ["Marketing", "Finance"],
            "desired_major": "Finance",
            "desired_city": "A",
        },
        {
            "category": 6,
            "number_of_people": 30,
            "suitable_majors": ["Finance"],
            "desired_major": "Finance",
            "desired_city": "B",
        },
    ]
    categories_C25 = [1, 2, 3, 4, 5, 6]
    # index 0 unused, categories 1..6 have 30 people
    category_sizes = [0, 30, 30, 30, 30, 30, 30]
    category_suitable_majors = {
        "1": ["Production", "Marketing"],
        "2": ["Marketing", "Finance"],
        "3": ["Production", "Finance"],
        "4": ["Production", "Finance"],
        "5": ["Marketing", "Finance"],
        "6": ["Finance"],
    }
    category_desired_major = {
        "1": "Production",
        "2": "Marketing",
        "3": "Production",
        "4": "Finance",
        "5": "Finance",
        "6": "Finance",
    }
    category_desired_city = {
        "1": "A",
        "2": "A",
        "3": "B",
        "4": "B",
        "5": "A",
        "6": "B",
    }

    # =========================
    # 2. Create model
    # =========================
    model = gp.Model("Rainbow_Group_Goal_Programming")

    # =========================
    # 3. Index sets
    # =========================
    I = range(1, i_range + 1)  # categories 1..6
    J = range(1, number_of_professions + 1)  # majors: 1=prod, 2=mkt, 3=fin
    K = range(1, 2 + 1)  # cities: 1=A, 2=B
    L = range(deviation_index_range_start, deviation_index_range_end + 1)  # 1..7

    # =========================
    # 4. Decision variables
    # =========================
    # x_{ijk} >= 0 integer
    x = model.addVars(I, J, K, vtype=gp.GRB.INTEGER, name="x")

    # Deviation variables d_l^+, d_l^- >= 0 (continuous)
    d_plus = model.addVars(L, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="d_plus")
    d_minus = model.addVars(L, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="d_minus")

    # =========================
    # 5. Constraints
    # =========================

    # 5.1 City–profession demand (Table C-24)
    # City A (k=1)
    model.addConstr(gp.quicksum(x[i, 1, 1] for i in I) == recruits_by_city_and_profession["A_Production"],
                    name="A_Production")
    model.addConstr(gp.quicksum(x[i, 2, 1] for i in I) == recruits_by_city_and_profession["A_Marketing"],
                    name="A_Marketing")
    model.addConstr(gp.quicksum(x[i, 3, 1] for i in I) == recruits_by_city_and_profession["A_Finance"],
                    name="A_Finance")

    # City B (k=2)
    model.addConstr(gp.quicksum(x[i, 1, 2] for i in I) == recruits_by_city_and_profession["B_Production"],
                    name="B_Production")
    model.addConstr(gp.quicksum(x[i, 2, 2] for i in I) == recruits_by_city_and_profession["B_Marketing"],
                    name="B_Marketing")
    model.addConstr(gp.quicksum(x[i, 3, 2] for i in I) == recruits_by_city_and_profession["B_Finance"],
                    name="B_Finance")

    # 5.2 Category capacity: each category has at most its size (30)
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, j, k] for j in J for k in K) <= category_sizes[i],
            name=f"CategoryCap_{i}"
        )

    # 5.3 Suitability constraints (hard constraints from Table C-25)
    # Category 1: suitable majors Production(1), Marketing(2) => no Finance(3)
    model.addConstr(x[1, 3, 1] == 0, name="Suit_1_Fin_A")
    model.addConstr(x[1, 3, 2] == 0, name="Suit_1_Fin_B")

    # Category 2: suitable majors Marketing(2), Finance(3) => no Production(1)
    model.addConstr(x[2, 1, 1] == 0, name="Suit_2_Prod_A")
    model.addConstr(x[2, 1, 2] == 0, name="Suit_2_Prod_B")

    # Category 3: suitable majors Production(1), Finance(3) => no Marketing(2)
    model.addConstr(x[3, 2, 1] == 0, name="Suit_3_Mkt_A")
    model.addConstr(x[3, 2, 2] == 0, name="Suit_3_Mkt_B")

    # Category 4: suitable majors Production(1), Finance(3) => no Marketing(2)
    model.addConstr(x[4, 2, 1] == 0, name="Suit_4_Mkt_A")
    model.addConstr(x[4, 2, 2] == 0, name="Suit_4_Mkt_B")

    # Category 5: suitable majors Marketing(2), Finance(3) => no Production(1)
    model.addConstr(x[5, 1, 1] == 0, name="Suit_5_Prod_A")
    model.addConstr(x[5, 1, 2] == 0, name="Suit_5_Prod_B")

    # Category 6: suitable major Finance(3) only => no Production(1), Marketing(2)
    model.addConstr(x[6, 1, 1] == 0, name="Suit_6_Prod_A")
    model.addConstr(x[6, 1, 2] == 0, name="Suit_6_Prod_B")
    model.addConstr(x[6, 2, 1] == 0, name="Suit_6_Mkt_A")
    model.addConstr(x[6, 2, 2] == 0, name="Suit_6_Mkt_B")

    # 5.4 Total recruitment goal (soft, as given): sum x + d_4^- - d_4^+ = 170
    total_recruits = gp.quicksum(x[i, j, k] for i in I for j in J for k in K)
    model.addConstr(
        total_recruits + d_minus[4] - d_plus[4] == total_recruits_planned,
        name="Goal_Total_Recruits"
    )

    # 5.5 Profession totals as goals (soft)
    # Production total (should be 20+25=45)
    total_production = gp.quicksum(x[i, 1, k] for i in I for k in K)
    model.addConstr(
        total_production + d_minus[5] - d_plus[5] == 45,
        name="Goal_Total_Production"
    )

    # Marketing total (should be 30+20=50)
    total_marketing = gp.quicksum(x[i, 2, k] for i in I for k in K)
    model.addConstr(
        total_marketing + d_minus[6] - d_plus[6] == 50,
        name="Goal_Total_Marketing"
    )

    # Finance total (should be 40+35=75)
    total_finance = gp.quicksum(x[i, 3, k] for i in I for k in K)
    model.addConstr(
        total_finance + d_minus[7] - d_plus[7] == 75,
        name="Goal_Total_Finance"
    )

    # 5.6 Goal 2: ≥80% of recruits in desired major
    # M = sum of recruits in their desired major (per category)
    M = (
        gp.quicksum(x[1, 1, k] for k in K) +  # cat 1, desired major: Production(1)
        gp.quicksum(x[2, 2, k] for k in K) +  # cat 2, desired major: Marketing(2)
        gp.quicksum(x[3, 1, k] for k in K) +  # cat 3, desired major: Production(1)
        gp.quicksum(x[4, 3, k] for k in K) +  # cat 4, desired major: Finance(3)
        gp.quicksum(x[5, 3, k] for k in K) +  # cat 5, desired major: Finance(3)
        gp.quicksum(x[6, 3, k] for k in K)    # cat 6, desired major: Finance(3)
    )
    # As specified in the validated model: M + d_2^- - d_2^+ = 136 (0.8*170)
    model.addConstr(
        M + d_minus[2] - d_plus[2] == int(p2_percentage_threshold * total_recruits_planned),
        name="Goal_Desired_Major"
    )

    # 5.7 Goal 3: ≥80% of recruits in desired city
    # C = sum of recruits in their desired city
    # Desired cities: cat1:A(1), cat2:A(1), cat3:B(2), cat4:B(2), cat5:A(1), cat6:B(2)
    C = (
        gp.quicksum(x[1, j, 1] for j in J) +  # category 1, city A(1)
        gp.quicksum(x[2, j, 1] for j in J) +  # category 2, city A(1)
        gp.quicksum(x[5, j, 1] for j in J) +  # category 5, city A(1)
        gp.quicksum(x[3, j, 2] for j in J) +  # category 3, city B(2)
        gp.quicksum(x[4, j, 2] for j in J) +  # category 4, city B(2)
        gp.quicksum(x[6, j, 2] for j in J)    # category 6, city B(2)
    )
    # As specified in the validated model: C + d_3^- - d_3^+ = 136
    model.addConstr(
        C + d_minus[3] - d_plus[3] == int(p3_percentage_threshold * total_recruits_planned),
        name="Goal_Desired_City"
    )

    # 5.8 Suitability goal p1
    # Suitability is enforced by zeroing unsuitable x_{ijk}, so d_1^± appear only in the objective.

    # =========================
    # 6. Objective function
    # =========================
    # As given: Min Z = 10000*(d_1^+ + d_1^-) + 100*(d_2^- + d_2^+) +
    #           1*(d_3^- + d_3^+ + d_4^- + d_4^+ + d_5^- + d_5^+ + d_6^- + d_6^+ + d_7^- + d_7^+)
    obj = (
        p1_weight * (d_plus[1] + d_minus[1]) +
        p2_weight * (d_minus[2] + d_plus[2]) +
        p3_weight * (
            d_minus[3] + d_plus[3] +
            d_minus[4] + d_plus[4] +
            d_minus[5] + d_plus[5] +
            d_minus[6] + d_plus[6] +
            d_minus[7] + d_plus[7]
        )
    )
    model.setObjective(obj, gp.GRB.MINIMIZE)

    # =========================
    # 7. Solve model
    # =========================
    model.optimize()

    # =========================
    # 8. Print results and FinalAnswer
    # =========================
    if model.status == gp.GRB.OPTIMAL:
        # Required: number of people in category 1, Production, City A => x_{1,1,1}
        x_1_1_1 = x[1, 1, 1].X

        # Print detailed result if desired
        print("Optimal objective value:", model.ObjVal)
        print("x_{1,1,1} (Category 1, Production, City A) =", int(round(x_1_1_1)))

        # FinalAnswer as required (unique solution)
        final_answer = int(round(x_1_1_1))
        print(f"FinalAnswer=【{final_answer}】")
    else:
        # If not optimal, still provide a FinalAnswer placeholder
        print("Model did not reach optimality. Status code:", model.status)
        print("FinalAnswer=【NaN】")


if __name__ == "__main__":
    solve_rainbow_group_goal_programming()
def optimize_candidate_selection(
    salaries={
        'F': 12000,
        'G': 15000,
        'H': 18000,
        'I': 5000,
        'J': 10000
    },
    skills={
        'F': 2,
        'G': 3,
        'H': 4,
        'I': 1,
        'J': 2
    },
    experiences={
        'F': 1,
        'G': 2,
        'H': 2,
        'I': 5,
        'J': 4
    },
    budget=40000,
    max_candidates=4,
    min_skill=8,
    min_experience=8,
    extra_cost_if_4=3000
):
    import gurobipy as gp
    from gurobipy import GRB

    # Initialize model
    model = gp.Model("Candidate_Selection")

    # Decision variables
    candidates = ['F', 'G', 'H', 'I', 'J']
    y = model.addVars(candidates, vtype=GRB.BINARY, name='hire')

    # Auxiliary binary variable: 1 if 4 candidates are hired, 0 otherwise
    z4 = model.addVar(vtype=GRB.BINARY, name='z_hire_4')

    # ❤ Non-linearity is introduced. ❤
    # Objective was: model.setObjective(gp.quicksum(salaries[c] * y[c] for c in candidates),
    # ❤ Non-linearity is introduced. ❤
    #            GRB.MINIMIZE)
    # New objective: total salaries + possible fixed cost if 4 people are hired
    model.setObjective(
        gp.quicksum(salaries[c] * y[c] for c in candidates) + extra_cost_if_4 * z4,
        GRB.MINIMIZE
    )

    # ❤ Non-linearity is introduced. ❤
    # Budget constraint was: model.addConstr(gp.quicksum(salaries[c] * y[c] for c in candidates)
    # ❤ Non-linearity is introduced. ❤
    #                             <= budget,
    #                             name='Budget')
    # New budget constraint: salaries + fixed management cost (if any) must satisfy budget
    model.addConstr(
        gp.quicksum(salaries[c] * y[c] for c in candidates) + extra_cost_if_4 * z4
        <= budget,
        name='Budget'
    )

    # Team size constraint: at most 4 candidates
    team_size_expr = gp.quicksum(y[c] for c in candidates)
    model.addConstr(team_size_expr <= max_candidates, name='MaxTeamSize')

    # Link z4 with team size:
    # If exactly 4 are hired then z4 = 1; otherwise z4 = 0.
    #
    # team_size_expr <= 3 + z4   (when z4 = 0, team_size <= 3; when z4 = 1, team_size <= 4)
    # team_size_expr >= 4 * z4   (when z4 = 1, team_size >= 4; when z4 = 0, team_size >= 0)
    model.addConstr(team_size_expr <= 3 + z4, name='Link_z4_upper')
    model.addConstr(team_size_expr >= 4 * z4, name='Link_z4_lower')

    # Skill level constraint
    model.addConstr(
        gp.quicksum(skills[c] * y[c] for c in candidates) >= min_skill,
        name='SkillLevel'
    )

    # Experience constraint
    model.addConstr(
        gp.quicksum(experiences[c] * y[c] for c in candidates) >= min_experience,
        name='Experience'
    )

    # Mutual exclusivity of G and J
    model.addConstr(y['G'] + y['J'] <= 1, name='G_J_Exclusion')

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        total_cost = model.objVal
        return total_cost
    else:
        return None


if __name__ == "__main__":
    result = optimize_candidate_selection()
    if result is not None:
        print(f"Optimal total cost (including possible management cost): {result}")
    else:
        print("No feasible solution found.")
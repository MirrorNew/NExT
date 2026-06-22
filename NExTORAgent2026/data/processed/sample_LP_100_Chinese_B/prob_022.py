def optimize_candidate_selection(salaries={
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
                                 min_experience=8):
    import gurobipy as gp
    from gurobipy import GRB

    # Initialize model
    model = gp.Model("Candidate_Selection")

    # Decision variables
    candidates = ['F', 'G', 'H', 'I', 'J']
    y = model.addVars(candidates, vtype=GRB.BINARY, name='hire')

    # Objective: Minimize total salary
    model.setObjective(gp.quicksum(salaries[c] * y[c] for c in candidates),
                       GRB.MINIMIZE)

    # Budget constraint
    model.addConstr(gp.quicksum(salaries[c] * y[c] for c in candidates)
                    <= budget,
                    name='Budget')

    # Team size constraint
    model.addConstr(gp.quicksum(y[c] for c in candidates) <= max_candidates,
                    name='MaxTeamSize')

    # Skill level constraint
    model.addConstr(gp.quicksum(skills[c] * y[c] for c in candidates)
                    >= min_skill,
                    name='SkillLevel')

    # ❤ Non-linearity is introduced. ❤
    # Experience constraint (original linear constraint):
    # model.addConstr(gp.quicksum(experiences[c] * y[c] for c in candidates)
    #                 >= min_experience,
    #                 name='Experience')
    # New non-linear experience constraint:
    #   total_experience / number_of_hires >= 2
    #   i.e., sum(experiences[c] * y[c]) / sum(y[c]) >= 2
    #   which is non-linear due to division by sum(y[c]).
    total_experience = gp.quicksum(experiences[c] * y[c] for c in candidates)
    team_size = gp.quicksum(y[c] for c in candidates)
    # Add non-linear constraint: total_experience / team_size >= 2
    # (Gurobi will treat this as a nonlinear constraint due to the division.)
    model.addConstr(total_experience / team_size >= 2, name='AvgExperience')

    # Mutual exclusivity of G and J
    model.addConstr(y['G'] + y['J'] <= 1, name='G_J_Exclusion')

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        total_salary = model.objVal
        return total_salary
    else:
        return None


if __name__ == "__main__":
    result = optimize_candidate_selection()
    if result is not None:
        print(f"Optimal total salary: {result}")
    else:
        print("No feasible solution found.")
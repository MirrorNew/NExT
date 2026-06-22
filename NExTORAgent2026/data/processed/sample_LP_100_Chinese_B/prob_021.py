import gurobipy as gp
from gurobipy import GRB


def solve_course_selection():
    """
    Solves the course selection problem to minimize the number of courses taken
    while satisfying category, prerequisite, and math-to-CS ratio requirements.
    """
    try:
        # --- Data ---
        courses = ['Calc', 'OR', 'DS', 'MS', 'Sim', 'Prog', 'Fcst']
        categories = ['Math', 'OpRes', 'CompSci']

        # Requirements per category
        category_requirements = {'Math': 2, 'OpRes': 2, 'CompSci': 2}

        # Course-to-category mapping: course_category_map[course][category] = 1 if belongs, else 0 or not present
        course_category_map = {
            'Calc': {
                'Math': 1
            },
            'OR': {
                'OpRes': 1,
                'Math': 1
            },
            'DS': {
                'CompSci': 1,
                'Math': 1
            },
            'MS': {
                'Math': 1,
                'OpRes': 1
            },
            'Sim': {
                'CompSci': 1,
                'OpRes': 1
            },
            'Prog': {
                'CompSci': 1
            },
            'Fcst': {
                'OpRes': 1,
                'Math': 1
            }
        }

        # Prerequisites: prereqs[course_requiring_prereq] = [list_of_prereqs]
        prerequisites = {
            'Sim': ['Prog'],
            'DS': ['Prog'],
            'MS': ['Calc'],
            'Fcst': ['MS']
        }

        # --- Create Gurobi Model ---
        model = gp.Model("CourseSelectionOptimization")

        # --- Decision Variables ---
        # x[c]: 1 if course c is selected, 0 otherwise
        x = model.addVars(courses, vtype=GRB.BINARY, name="x")

        # --- Objective Function: Minimize the total number of courses selected ---
        model.setObjective(gp.quicksum(x[c] for c in courses), GRB.MINIMIZE)

        # --- Constraints ---
        # 1. Category Requirements
        for cat in categories:
            model.addConstr(
                gp.quicksum(x[c] * course_category_map[c].get(cat, 0) for c in courses)
                >= category_requirements[cat],
                name=f"Req_{cat}"
            )

        # 2. Prerequisite Constraints
        # If x[course_with_prereq] = 1, then x[prereq_course] must be 1.
        # So, x[course_with_prereq] <= x[prereq_course]
        for course, prereq_list in prerequisites.items():
            for prereq_course in prereq_list:
                if course in x and prereq_course in x:  # Ensure both courses are valid keys
                    model.addConstr(
                        x[course] <= x[prereq_course],
                        name=f"Prereq_{prereq_course}_for_{course}"
                    )

        # 3. Math / CompSci ratio constraint: Math_count / CS_count <= 4/3
        #    i.e., 3 * Math_count <= 4 * CS_count
        #    Math_count = sum_c x[c] * 1{c in Math}
        #    CS_count   = sum_c x[c] * 1{c in CompSci}

        # ❤ Non-linearity is introduced. ❤
        math_count = gp.quicksum(
            x[c] * course_category_map[c].get('Math', 0) for c in courses
        )
        comp_sci_count = gp.quicksum(
            x[c] * course_category_map[c].get('CompSci', 0) for c in courses
        )
        model.addConstr(
            3 * math_count <= 4 * comp_sci_count,
            name="Math_to_CS_ratio"
        )

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal course selection found.")
            print(f"Minimum number of courses to take: {model.ObjVal:.0f}")

            print("\nSelected Courses:")
            selected_courses_list = []
            for c in courses:
                if x[c].X > 0.5:  # If x[c] is 1
                    selected_courses_list.append(c)
                    print(f"  - {c}")

            print("\nVerification of Category Requirements:")
            for cat in categories:
                courses_for_cat = 0
                cat_courses_taken = []
                for c_taken in selected_courses_list:
                    if course_category_map[c_taken].get(cat, 0) == 1:
                        courses_for_cat += 1
                        cat_courses_taken.append(c_taken)
                print(
                    f"  Category '{cat}': Required={category_requirements[cat]}, "
                    f"Taken={courses_for_cat} ({', '.join(cat_courses_taken)})"
                )

            print("\nVerification of Prerequisites:")
            all_prereqs_met = True
            for course_taken in selected_courses_list:
                if course_taken in prerequisites:
                    for prereq_c in prerequisites[course_taken]:
                        if prereq_c not in selected_courses_list:
                            print(
                                f"  ERROR: Course '{course_taken}' taken, "
                                f"but its prerequisite '{prereq_c}' is NOT taken."
                            )
                            all_prereqs_met = False
            if all_prereqs_met:
                print(
                    "  All prerequisite conditions are met for the selected courses."
                )

            # Verification of Math/CompSci ratio
            print("\nVerification of Math/CompSci Ratio (Math_count / CS_count <= 4/3):")
            math_taken = sum(
                course_category_map[c].get('Math', 0) for c in selected_courses_list
            )
            cs_taken = sum(
                course_category_map[c].get('CompSci', 0) for c in selected_courses_list
            )
            print(f"  Math courses counted (by category membership): {math_taken}")
            print(f"  CompSci courses counted (by category membership): {cs_taken}")
            if cs_taken > 0:
                print(f"  Ratio Math/CompSci = {math_taken}/{cs_taken} = {math_taken / cs_taken:.3f}")
            else:
                print("  WARNING: No CompSci course counted; ratio is undefined in this check.")

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. The requirements cannot be met with the given courses and constraints."
            )
            # Compute and print IIS (Irreducible Inconsistent Subsystem) if needed
            # model.computeIIS()
            # model.write("course_selection_iis.ilp")
            # print("IIS written to course_selection_iis.ilp.")
        else:
            print(f"Optimization stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_course_selection()
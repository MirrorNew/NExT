import gurobipy as gp
from gurobipy import GRB


def solve_course_selection():
    """
    Solves the course selection problem to minimize the number of courses taken
    while satisfying category, prerequisite requirements, and the new rule
    about the extra 'AcademicWriting' course when more than 5 main courses
    are selected.
    """
    try:
        # --- Data ---
        # Original seven main courses
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
            # Note: 'AcademicWriting' will NOT be added here, so it does not
            # count toward any category.
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
        # x[c]: 1 if main course c is selected, 0 otherwise
        x = model.addVars(courses, vtype=GRB.BINARY, name="x")

        # Additional binary variable for the extra seminar course "AcademicWriting"
        # It does not belong to any category and has no prerequisites.
        x_aw = model.addVar(vtype=GRB.BINARY, name="x_AcademicWriting")

        # Auxiliary binary variable to represent:
        # y = 1 if total number of main courses selected > 5 (i.e., 6 or 7), else 0.
        y = model.addVar(vtype=GRB.BINARY, name="y_over5_main")

        # --- Objective Function: Minimize the total number of ALL courses selected ---
        # ❤ Non-linearity is introduced. ❤
        # model.setObjective(gp.quicksum(x[c] for c in courses), GRB.MINIMIZE)
        model.setObjective(gp.quicksum(x[c] for c in courses) + x_aw, GRB.MINIMIZE)

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

        # 3. New non-linear rule (modeled linearly with binary variable y and big-M):
        #    If the total number of MAIN courses selected > 5 (i.e., 6 or 7),
        #    then AcademicWriting must be selected (x_aw = 1).
        #    If total main courses <= 5, then AcademicWriting must NOT be selected (x_aw = 0).
        #
        # Let total_main = sum_c x[c].
        # We want:
        #   total_main > 5  -> x_aw = 1
        #   total_main <= 5 -> x_aw = 0
        #
        # Use binary y to represent "total_main > 5":
        #   total_main >= 6*y
        #   total_main <= 5 + 2*y    (since max total_main is 7)
        # Then:
        #   x_aw = y
        total_main = gp.quicksum(x[c] for c in courses)

        # total_main >= 6*y
        model.addConstr(total_main >= 6 * y, name="link_y_lower")

        # total_main <= 5 + 2*y   (when y=0 -> total_main<=5, when y=1 -> total_main<=7)
        model.addConstr(total_main <= 5 + 2 * y, name="link_y_upper")

        # x_aw must equal y:
        model.addConstr(x_aw - y == 0, name="link_aw_y")

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal course selection found.")
            print(f"Minimum number of courses to take (including AcademicWriting if required): {model.ObjVal:.0f}")

            print("\nSelected Courses:")
            selected_courses_list = []
            for c in courses:
                if x[c].X > 0.5:  # If x[c] is 1
                    selected_courses_list.append(c)
                    print(f"  - {c}")
            if x_aw.X > 0.5:
                print("  - AcademicWriting")
                selected_courses_list.append("AcademicWriting")

            print("\nVerification of Category Requirements (main courses only):")
            for cat in categories:
                courses_for_cat = 0
                cat_courses_taken = []
                for c_taken in courses:  # only the 7 main courses have categories
                    if x[c_taken].X > 0.5 and course_category_map[c_taken].get(cat, 0) == 1:
                        courses_for_cat += 1
                        cat_courses_taken.append(c_taken)
                print(
                    f"  Category '{cat}': Required={category_requirements[cat]}, "
                    f"Taken={courses_for_cat} ({', '.join(cat_courses_taken)})"
                )

            print("\nVerification of Prerequisites (main courses only):")
            all_prereqs_met = True
            for course_taken in courses:
                if x[course_taken].X > 0.5 and course_taken in prerequisites:
                    for prereq_c in prerequisites[course_taken]:
                        if x[prereq_c].X < 0.5:
                            print(
                                f"  ERROR: Course '{course_taken}' taken, "
                                f"but its prerequisite '{prereq_c}' is NOT taken."
                            )
                            all_prereqs_met = False
            if all_prereqs_met:
                print("  All prerequisite conditions are met for the selected courses.")

            print("\nVerification of AcademicWriting rule:")
            total_main_val = sum(1 for c in courses if x[c].X > 0.5)
            aw_taken = x_aw.X > 0.5
            print(f"  Main courses selected: {total_main_val}")
            print(f"  AcademicWriting selected: {aw_taken}")
            if total_main_val > 5 and not aw_taken:
                print("  ERROR: More than 5 main courses selected but AcademicWriting is NOT taken.")
            elif total_main_val <= 5 and aw_taken:
                print("  ERROR: 5 or fewer main courses selected but AcademicWriting IS taken.")
            else:
                print("  AcademicWriting rule is satisfied.")

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
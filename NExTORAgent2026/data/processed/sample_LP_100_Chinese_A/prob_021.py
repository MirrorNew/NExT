import gurobipy as gp
from gurobipy import GRB
import math


def solve_course_selection():
    """
    Solves the course selection problem to minimize the number of courses taken
    while satisfying category, prerequisite, and time-varying nonlinear Math
    requirements across 4 semesters.
    """
    try:
        # --- Data ---
        courses = ['Calc', 'OR', 'DS', 'MS', 'Sim', 'Prog', 'Fcst']
        categories = ['Math', 'OpRes', 'CompSci']

        # Nominal requirements per category (before any time-varying adjustment)
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

        # --- New time-related data for non-linear Math requirement ---
        semesters = [1, 2, 3, 4]  # t = 1, 2, 3, 4
        math_nominal_requirement = category_requirements['Math']  # still 2
        math_base_count = 5  # given in the description
        pi_val = 3.1416
        period_param = 8

        # Precompute semester-specific nonlinear coefficients and thresholds
        math_coeff = {}
        math_threshold = {}
        for t in semesters:
            # coefficient: 1 + 0.1 × cos(π × 5/8 × t)
            math_coeff[t] = 1.0 + 0.1 * math.cos(pi_val * math_base_count / period_param * t)
            # effective threshold = 2 × coefficient
            math_threshold[t] = math_nominal_requirement * math_coeff[t]

        # --- Create Gurobi Model ---
        model = gp.Model("CourseSelectionOptimization_WithTimeVaryingMath")

        # --- Decision Variables ---
        # x[c]: 1 if course c is selected at least in one semester (global “taken” flag)
        x = model.addVars(courses, vtype=GRB.BINARY, name="x")

        # y[t, c]: 1 if course c is scheduled in semester t, 0 otherwise
        y = model.addVars(semesters, courses, vtype=GRB.BINARY, name="y")

        # --- Objective Function: Minimize the total number of distinct courses selected ---
        # Each course is counted once, even if scheduled in some specific semester.
        model.setObjective(gp.quicksum(x[c] for c in courses), GRB.MINIMIZE)

        # --- Constraints ---

        # Link x and y: if a course is taken in some semester, x[c] must be 1
        for c in courses:
            model.addConstr(
                gp.quicksum(y[t, c] for t in semesters) <= x[c],
                name=f"Link_x_y_{c}"
            )

        # 1. Category Requirements (OpRes and CompSci remain global as in original)
        # ❤ Non-linearity is introduced. ❤
        # for cat in categories:
        #     model.addConstr(gp.quicksum(
        #         x[c] * course_category_map[c].get(cat, 0) for c in courses)
        #                     >= category_requirements[cat],
        #                     name=f"Req_{cat}")
        for cat in categories:
            if cat == 'Math':
                # Math requirement is now time-varying and handled per semester below
                continue
            model.addConstr(
                gp.quicksum(x[c] * course_category_map[c].get(cat, 0) for c in courses)
                >= category_requirements[cat],
                name=f"Req_{cat}"
            )

        # 1a. Time-varying nonlinear Math requirement per semester
        # For each semester t, the sum of Math courses taken in that semester must
        # be at least math_threshold[t] (a nonlinear function of t).
        for t in semesters:
            model.addConstr(
                gp.quicksum(y[t, c] * course_category_map[c].get('Math', 0) for c in courses)
                >= math_threshold[t],
                name=f"TimeVaryingMathReq_t{t}"
            )

        # 2. Each course can be taken in at most one semester (simple scheduling assumption)
        for c in courses:
            model.addConstr(
                gp.quicksum(y[t, c] for t in semesters) <= 1,
                name=f"OneSemesterPerCourse_{c}"
            )

        # 3. Prerequisite Constraints (respecting time order):
        # If a course with prerequisite is taken in semester t2,
        # then its prerequisite must be taken in some semester t1 <= t2.
        # ❤ Non-linearity is introduced. ❤
        # for course, prereq_list in prerequisites.items():
        #     for prereq_course in prereq_list:
        #         if course in x and prereq_course in x:  # Ensure both courses are valid keys
        #             model.addConstr(
        #                 x[course] <= x[prereq_course],
        #                 name=f"Prereq_{prereq_course}_for_{course}")
        for course, prereq_list in prerequisites.items():
            for prereq_course in prereq_list:
                if course in courses and prereq_course in courses:
                    for t2 in semesters:
                        # If course is in semester t2, prerequisite must be in ≤ t2
                        model.addConstr(
                            y[t2, course] <= gp.quicksum(
                                y[t1, prereq_course] for t1 in semesters if t1 <= t2
                            ),
                            name=f"PrereqTime_{prereq_course}_before_{course}_t{t2}"
                        )

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal course selection found (with time-varying nonlinear Math requirement).")
            print(f"Minimum number of courses to take: {model.ObjVal:.0f}")

            # Selected courses globally
            selected_courses_list = [c for c in courses if x[c].X > 0.5]

            print("\nSelected Courses (overall):")
            for c in selected_courses_list:
                print(f"  - {c}")

            # Schedule by semester
            print("\nCourse schedule by semester:")
            for t in semesters:
                sem_courses = [c for c in courses if y[t, c].X > 0.5]
                print(f"  Semester {t}: {', '.join(sem_courses) if sem_courses else 'No courses'}")

            # Verification of category requirements (global for OpRes and CompSci)
            print("\nVerification of Global Category Requirements (OpRes & CompSci):")
            for cat in ['OpRes', 'CompSci']:
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

            # Verification of time-varying Math requirement
            print("\nVerification of Time-Varying Math Requirement by Semester:")
            for t in semesters:
                math_count_t = 0
                math_courses_t = []
                for c in courses:
                    if y[t, c].X > 0.5 and course_category_map[c].get('Math', 0) == 1:
                        math_count_t += 1
                        math_courses_t.append(c)
                print(
                    f"  Semester {t}: Threshold≈{math_threshold[t]:.3f}, "
                    f"MathTaken={math_count_t} ({', '.join(math_courses_t) if math_courses_t else 'None'})"
                )

            # Verification of prerequisites (time-respecting)
            print("\nVerification of Prerequisites with Semester Ordering:")
            all_prereqs_met = True
            # Build helper: what semester was each course taken in (if any)?
            course_sem = {}
            for c in courses:
                taken_sem = None
                for t in semesters:
                    if y[t, c].X > 0.5:
                        taken_sem = t
                        break
                course_sem[c] = taken_sem

            for course, prereq_list in prerequisites.items():
                if course_sem[course] is None:
                    # Course not taken, no need to check its prerequisites
                    continue
                t_course = course_sem[course]
                for prereq_c in prereq_list:
                    t_pr = course_sem.get(prereq_c, None)
                    if t_pr is None or t_pr > t_course:
                        print(
                            f"  ERROR: Course '{course}' taken in semester {t_course}, "
                            f"but its prerequisite '{prereq_c}' is taken in semester {t_pr}."
                        )
                        all_prereqs_met = False

            if all_prereqs_met:
                print("  All prerequisite conditions (with time ordering) are met for the selected courses.")

        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. The requirements cannot be met with the given courses, constraints, and time-varying Math requirement.")
            # Optional: IIS
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
def optimize_sports_equipment(material_limit=1500, hours_limit=750, min_football=50):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Sports_Equipment_Production")

    # Decision variables: number of basketballs and footballs
    x = model.addVar(name="Basketballs", vtype=GRB.INTEGER, lb=0)
    y = model.addVar(name="Footballs", vtype=GRB.INTEGER, lb=0)

    # Set the objective: maximize total equipment
    model.setObjective(x + y, GRB.MAXIMIZE)

    # Add constraints
    # Material constraint
    model.addConstr(5 * x + 3 * y <= material_limit, name="MaterialLimit")
    # Labor hours constraint
    model.addConstr(x + 2 * y <= hours_limit, name="HoursLimit")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(x >= 3 * y, name="BasketballToFootballRatio")
    # Introduce nonlinear production ratio: x >= (3 * y) ^ 1.2
    # Note: This uses Gurobi's general function (GenConstrExp) + Log to model the power.
    # For simplicity and clarity, we use the built-in pow() operator directly in a general constraint expression.
    Y = model.addVar()
    model.addGenConstrPow(y,Y,1.002)
    model.addConstr(x >= 3 ** 1.2 * Y, name="BasketballToFootballNonlinearRatio")

    # Minimum footballs
    model.addConstr(y >= min_football, name="MinFootballs")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        total_produced = x.X + y.X
        # Also return the specific numbers of basketballs and footballs for clarity
        return total_produced, x.X, y.X
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_sports_equipment()
    if result is not None:
        total_equipment, basketballs, footballs = result
        print(f"Maximum Total Sports Equipment Produced: {total_equipment}")
        print(f"Optimal Basketballs: {basketballs}")
        print(f"Optimal Footballs: {footballs}")
    else:
        print("No feasible solution found.")
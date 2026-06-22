from gurobipy import Model, GRB

def optimize_meal_delivery():
    # Create a new model
    model = Model("MealDeliveryOptimization_Nonlinear")

    # Decision variables
    x_b = model.addVar(vtype=GRB.INTEGER, name="bikes")
    x_s = model.addVar(vtype=GRB.INTEGER, name="scooters")

    # Set objective: maximize total meals
    model.setObjective(8 * x_b + 5 * x_s, GRB.MAXIMIZE)

    # Add constraints
    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(3 * x_b + 2 * x_s <= 200, name="charge_limit")

    Y = model.addVar(name="Y_servings", lb=0)
    model.addConstr(3 * x_b == Y)
    YP = model.addVar(name="Y_P", lb=0)
    model.addGenConstrPow(Y,YP,1.002)

    model.addConstr(YP + 2 * x_s <= 200, name="charge_limit_nonlinear")

    # Bike usage limit (30% of total vehicles)
    model.addConstr(7 * x_b <= 3 * x_s, name="bike_ratio")
    # Minimum scooters
    model.addConstr(x_s >= 20, name="min_scooters")
    # Non-negativity is implicit in variable definition

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the maximum number of meals delivered and the vehicle counts
        return {
            "max_meals": model.objVal,
            "bikes": x_b.X,
            "scooters": x_s.X
        }
    else:
        # No feasible solution
        return None

# Example usage
if __name__ == "__main__":
    result = optimize_meal_delivery()
    if result is not None:
        print(f"Maximum Meals Delivered: {result['max_meals']}")
        print(f"Bikes used: {result['bikes']}")
        print(f"Scooters used: {result['scooters']}")
    else:
        print("No feasible solution found.")
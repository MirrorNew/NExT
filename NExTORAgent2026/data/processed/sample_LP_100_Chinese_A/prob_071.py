import gurobipy as gp
from gurobipy import GRB

def optimize_transportation(
    capacity_truck=40,     # capacity of a tractor in kg (base, before nonlinear correction)
    capacity_car=20,       # capacity of a car in kg
    min_corn=500,          # minimum kg of corn to send
    ratio_cars_to_tractors=2  # cars >= 2 * tractors
):
    # Create a new model
    model = gp.Model("CornTransportOptimization_Nonlinear")

    # Enable non-convex quadratic / polynomial constraints
    # (required because we will introduce a nonlinear term)
    model.Params.NonConvex = 2

    # Decision variables: number of tractors and cars
    x = model.addVar(vtype=GRB.INTEGER, name="tractors", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="cars", lb=0)

    # Set objective: minimize total number of vehicles
    model.setObjective(x + y, GRB.MINIMIZE)

    # Capacity constraint: total transported >= min_corn
    # 40 * x (tractor part) is replaced by (40 * x)^1.2
    # ❤ Non-linearity is introduced. ❤
    Y = model.addVar()
    model.addConstr(Y==capacity_truck * x)
    YY = model.addVar()
    model.addGenConstrPow(Y,YY,1.02)
    model.addConstr(YY + capacity_car * y >= min_corn, name="capacity")

    # Ratio constraint: cars >= 2 * tractors
    model.addConstr(y >= ratio_cars_to_tractors * x, name="ratio")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the minimal total number of vehicles
        return model.objVal, x.X, y.X
    else:
        # No feasible solution
        return None, None, None

# Example usage
if __name__ == "__main__":
    min_vehicles, tractors, cars = optimize_transportation()
    if min_vehicles is not None:
        print(f"Minimum Total Vehicles: {min_vehicles}")
        print(f"Tractors used: {tractors}")
        print(f"Cars used: {cars}")
    else:
        print("No feasible solution found.")
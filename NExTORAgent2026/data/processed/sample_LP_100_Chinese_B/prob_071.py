import gurobipy as gp
from gurobipy import GRB

def optimize_transportation(
    capacity_truck=40,     # capacity of a tractor in kg
    capacity_car=20,       # capacity of a car in kg
    min_corn=500,          # minimum kg of corn to send
    ratio_cars_to_tractors=2  # cars / tractors >= 2
):
    # Create a new model
    model = gp.Model("CornTransportOptimization")
    
    # Decision variables: number of tractors and cars
    x = model.addVar(vtype=GRB.CONTINUOUS, name="tractors", lb=0)
    y = model.addVar(vtype=GRB.CONTINUOUS, name="cars", lb=0)
    
    # Set objective: minimize total number of vehicles
    model.setObjective(x + y, GRB.MINIMIZE)
    
    # Capacity constraint: total transported >= min_corn
    model.addConstr(capacity_truck * x + capacity_car * y >= min_corn, name="capacity")
    
    # ❤ Non-linearity is introduced. ❤
    # Original linear constraint (commented out):
    # model.addConstr(y >= ratio_cars_to_tractors * x, name="ratio")
    # Non-linear ratio constraint: y / x >= 2  (for x > 0)
    # Implemented as a quadratic constraint: y * x >= 2 * x * x
    model.addQConstr(y * x >= ratio_cars_to_tractors * x * x, name="nonlinear_ratio")

    # Optimize the model
    model.optimize()
    
    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the minimal total number of vehicles
        return model.objVal
    else:
        # No feasible solution
        return None

# Example usage
if __name__ == "__main__":
    min_vehicles = optimize_transportation()
    if min_vehicles is not None:
        print(f"Minimum Total Vehicles: {min_vehicles}")
    else:
        print("No feasible solution found.")
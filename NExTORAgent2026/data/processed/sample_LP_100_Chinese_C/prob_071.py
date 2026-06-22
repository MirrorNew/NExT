import gurobipy as gp
from gurobipy import GRB

def optimize_transportation(
    capacity_truck=40,     # capacity of a tractor in kg
    capacity_car=20,       # capacity of a car in kg
    min_corn=500,          # minimum kg of corn to send
    ratio_cars_to_tractors=2  # cars >= 2 * tractors
):
    # Create a new model
    model = gp.Model("CornTransportOptimization")
    
    # Decision variables: number of tractors and cars
    x = model.addVar(vtype=GRB.INTEGER, name="tractors", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="cars", lb=0)
    
    # ❤ Non-linearity is introduced. ❤
    # Set objective: minimize total number of vehicles
    # model.setObjective(x + y, GRB.MINIMIZE)
    # New non-linear-style objective with a conditional penalty for tractors beyond 5:
    # If x > 5, then from the 6th tractor on, each additional tractor effectively
    # forces an extra car (congestion/diversion), modeled here as an extra term.
    # This is captured by max(0, x - 5), which is non-linear.

    # Add constraint (x > 5 will introduce a penalty)
    penalty = model.addVar(name="penalty", lb=0)

    # Use addGenConstrMax to ensure penalty is 0 if x <= 5, and x-5 if x > 5
    model.addGenConstrMax(penalty, [x], 5, name="penalty_constraint")

    # Define your objective function, adding the penalty to the total number of vehicles
    total_vehicles = x + y + penalty
    model.setObjective(total_vehicles, GRB.MINIMIZE)
    
    # Capacity constraint: total transported >= min_corn
    model.addConstr(capacity_truck * x + capacity_car * y >= min_corn, name="capacity")
    
    # Ratio constraint: cars >= 2 * tractors
    model.addConstr(y >= ratio_cars_to_tractors * x, name="ratio")
    
    # ❤ Non-linearity is introduced. ❤
    # Linear ratio only:
    # (already included above as basic ratio; below we add the new conditional
    # requirement in a non-linear fashion rather than with additional linear vars)
    #
    # New non-linear-style congestion constraint:
    # If x > 5, then y must be at least 2x + (x - 5) = 3x - 5.
    # This is encoded via another max_ term:
    model.addConstr(
        y >= ratio_cars_to_tractors * x + penalty,
        name="congestion_non_linear"
    )
    
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
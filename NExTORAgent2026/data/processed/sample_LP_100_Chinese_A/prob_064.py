from gurobipy import Model, GRB

def optimize_pipes():
    # Create a new model
    m = Model("PipeOptimization")

    # Decision variables: number of high-volume and low-volume pipes
    x = m.addVar(vtype=GRB.INTEGER, name="HighVolumePipes")
    y = m.addVar(vtype=GRB.INTEGER, name="LowVolumePipes")

    # Auxiliary variable for nonlinear term y^1.2
    y_eff = m.addVar(vtype=GRB.CONTINUOUS, name="LowVolumePipes_Effective")

    # Set the objective: minimize total number of pipes
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add demand constraint
    m.addConstr(10000 * x + 5000 * y >= 150000, name="DemandConstraint")

    # Add staffing constraint
    m.addConstr(12 * x + 5 * y <= 160, name="StaffingConstraint")

    # Add high-volume pipe proportion constraint: x ≤ 35% of total pipes
    m.addConstr(x <= 0.35 * (x + y), name="HighVolumeProportion")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(y >= 8, name="MinLowVolumePipes")

    # Nonlinear minimum low-volume pipes constraint: y^1.2 ≥ 8
    # Implemented using a general constraint with a power function
    m.addGenConstrPow(y, y_eff, 0.65, name="LowVolumePipesPower")
    m.addConstr(y_eff >= 8, name="MinEffectiveLowVolumePipes")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of pipes and the breakdown
        return {
            "TotalPipes": m.objVal,
            "HighVolumePipes": x.X,
            "LowVolumePipes": y.X
        }
    else:
        # No feasible solution found
        return None

# Example usage
if __name__ == "__main__":
    result = optimize_pipes()
    if result is not None:
        print(f"Minimum Total Pipes: {result['TotalPipes']}")
        print(f"High Volume Pipes: {result['HighVolumePipes']}")
        print(f"Low Volume Pipes: {result['LowVolumePipes']}")
    else:
        print("No feasible solution found.")
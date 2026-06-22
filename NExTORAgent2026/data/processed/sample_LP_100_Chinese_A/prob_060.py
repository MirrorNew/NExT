import gurobipy as gp
from gurobipy import GRB

def optimize_supplements(
    cost_A=2,        # Cost per pill of supplement A
    cost_B=3,        # Cost per pill of supplement B
    iron_A=5,        # Iron units in supplement A
    calcium_A=10,    # Calcium units in supplement A
    iron_B=4,        # Iron units in supplement B
    calcium_B=15,    # Calcium units in supplement B
    min_iron=40,     # Minimum iron requirement
    min_calcium=50,  # Minimum calcium requirement
    k=0.01            # Interaction coefficient for non-linear cost
):
    # Create a new model
    model = gp.Model("SupplementsOptimization_Nonlinear")

    # Add decision variables (integer, non-negative)
    x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_A")
    x_B = model.addVar(vtype=GRB.INTEGER, lb=1, name="x_B")

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(cost_A * x_A + cost_B * x_B, GRB.MINIMIZE)

    # Add non-linear expression for the interaction cost: k * x * y * (x + y)
    # Here we model the full objective: 2 * x_A + 3 * x_B + 0.1 * x_A * x_B * (x_A + x_B)

    Y = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Y")
    interaction_cost = k * Y * (x_A + x_B)
    model.addConstr(Y == x_A * x_B)
    model.setObjective(cost_A * x_A + cost_B * x_B + interaction_cost, GRB.MINIMIZE)

    # Add constraints
    # Iron constraint
    model.addConstr(iron_A * x_A + iron_B * x_B >= min_iron, "IronRequirement")
    # Calcium constraint
    model.addConstr(calcium_A * x_A + calcium_B * x_B >= min_calcium, "CalciumRequirement")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the optimal decision variables and total cost
        return {
            "x_A": x_A.X,
            "x_B": x_B.X,
            "min_cost": model.objVal
        }
    else:
        # No feasible solution found
        return None

# Example usage
if __name__ == "__main__":
    result = optimize_supplements()
    if result is not None:
        print(f"Optimal number of pills of supplement A (x): {result['x_A']}")
        print(f"Optimal number of pills of supplement B (y): {result['x_B']}")
        print(f"Minimum total cost (including interaction): {result['min_cost']}")
    else:
        print("No feasible solution found.")
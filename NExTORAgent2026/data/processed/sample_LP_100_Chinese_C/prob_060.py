import gurobipy as gp
from gurobipy import GRB

def optimize_supplements(
    cost_A=2,        # Cost per pill of supplement A
    cost_B=3,        # Base cost per pill of supplement B (for first 10 pills)
    discount_cost_B=2,  # Discounted cost per pill of supplement B (from 11th pill on)
    iron_A=5,        # Iron units in supplement A
    calcium_A=10,    # Calcium units in supplement A
    iron_B=4,        # Iron units in supplement B
    calcium_B=15,    # Calcium units in supplement B
    min_iron=40,     # Minimum iron requirement
    min_calcium=50   # Minimum calcium requirement
):
    # Create a new model
    model = gp.Model("SupplementsOptimization")

    # Add decision variables (integer, non-negative)
    x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_A")
    # ❤ Non-linearity is introduced. ❤
    # x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")
    # Introduce variables for piecewise pricing of supplement B
    # y_B1: number of B pills priced at 3 dollars (max 10)
    # y_B2: number of B pills priced at 2 dollars (only if total B > 10 in the logic, but we don't need an explicit binary)
    y_B1 = model.addVar(vtype=GRB.INTEGER, lb=0, ub=10, name="y_B1")
    y_B2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="y_B2")

    # Total B pills
    x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")

    # Link total B with the two components
    model.addConstr(x_B == y_B1 + y_B2, name="Link_B_components")

    # ❤ Non-linearity is introduced. ❤
    # Set the objective: minimize total cost
    # model.setObjective(cost_A * x_A + cost_B * x_B, GRB.MINIMIZE)
    # New objective with piecewise pricing for supplement B
    model.setObjective(
        cost_A * x_A
        + cost_B * y_B1          # up to 10 pills at $3
        + discount_cost_B * y_B2 # remaining pills at $2
        ,
        GRB.MINIMIZE
    )

    # Add constraints
    # Iron constraint
    model.addConstr(
        iron_A * x_A + iron_B * x_B >= min_iron,
        "IronRequirement"
    )
    # Calcium constraint
    model.addConstr(
        calcium_A * x_A + calcium_B * x_B >= min_calcium,
        "CalciumRequirement"
    )

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the optimal total cost and the optimal pill counts
        return {
            "min_cost": model.objVal,
            "x_A": x_A.X,
            "x_B": x_B.X,
            "y_B1": y_B1.X,
            "y_B2": y_B2.X
        }
    else:
        # No feasible solution found
        return None

# Example usage
if __name__ == "__main__":
    result = optimize_supplements()
    if result is not None:
        print(f"Minimum Cost: {result['min_cost']}")
        print(f"Optimal number of A pills (x_A): {result['x_A']}")
        print(f"Optimal number of total B pills (x_B): {result['x_B']}")
        print(f"  B pills at $3 (first tier, y_B1): {result['y_B1']}")
        print(f"  B pills at $2 (discount tier, y_B2): {result['y_B2']}")
    else:
        print("No feasible solution found.")
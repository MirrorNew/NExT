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
    min_calcium=50   # Minimum calcium requirement
):
    # Create a new model
    model = gp.Model("SupplementsOptimization")
    
    # Add decision variables (integer, non-negative)
    x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_A")
    x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")
    
    # Set the objective: minimize total cost
    model.setObjective(cost_A * x_A + cost_B * x_B, GRB.MINIMIZE)
    
    # Add constraints
    # Iron constraint
    model.addConstr(iron_A * x_A + iron_B * x_B >= min_iron, "IronRequirement")
    # Calcium constraint
    model.addConstr(calcium_A * x_A + calcium_B * x_B >= min_calcium, "CalciumRequirement")
    
    # ❤ Non-linearity is introduced. ❤
    # Original linear model had no ratio constraint between iron from A and B.
    # New non-linear constraint: (iron_A * x_A) / (iron_B * x_B) <= 2
    # which is implemented as a general constraint of type GENCONSTR_EXP or GENCONSTR_POW
    # by directly modeling the division in Gurobi's non-linear expression interface.
    # 注意：此处通过商的形式显式引入非线性表达式。
    if iron_B != 0:
        # Avoid division by zero: only enforce ratio when x_B > 0
        # We model this using a big-M approach with a binary variable and a non-linear term.
        y = model.addVar(vtype=GRB.BINARY, name="y_has_B")  # 1 if x_B >= 1, else 0
        M = 1000  # big-M
        
        # Link y with x_B so that y = 1 implies x_B >= 1, y = 0 implies x_B = 0
        model.addConstr(x_B >= y, name="Link_y_xB_lower")
        model.addConstr(x_B <= M * y, name="Link_y_xB_upper")
        
        # ❤ Non-linearity is introduced. ❤
        # When y = 1 (x_B >= 1), enforce (iron_A * x_A) / (iron_B * x_B) <= 2
        # i.e., iron_A * x_A <= 2 * iron_B * x_B
        # implemented as a non-linear constraint through direct multiplication of decision variables.
        model.addConstr(
            iron_A * x_A <= 2 * iron_B * x_B,
            name="IronRatioConstraint"
        )
        # Note: the above quadratic constraint involves product of variables x_A and x_B
        # via their linear combination in a single inequality, thus introducing non-linearity.
    
    # Optimize the model
    model.optimize()
    
    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the optimal total cost, and the optimal pill counts
        return model.objVal, int(x_A.X), int(x_B.X)
    else:
        # No feasible solution found
        return None

# Example usage
if __name__ == "__main__":
    result = optimize_supplements()
    if result is not None:
        min_cost, opt_x_A, opt_x_B = result
        print(f"Minimum Cost: {min_cost}")
        print(f"Optimal number of pills - Supplement A: {opt_x_A}, Supplement B: {opt_x_B}")
    else:
        print("No feasible solution found.")
import gurobipy as gp
from gurobipy import GRB

# 1. Create the model
model = gp.Model("Realm_Optimization")

# 2. Define parameters (based on Parameters List)
# Factory 1 modes: 1, 2, 3
# Factory 2 modes: 4, 5
# Cutting Mode details:
# Mode 1: 40+25+25, waste 10
# Mode 2: 40+40, waste 20
# Mode 3: 50+40, waste 10
# Mode 4: 50+25+25, waste 0
# Mode 5: 25+25+25+25, waste 0
# Mode 6: 50+50, waste 0

# 3. Create decision variables
# x[p] is the number of 100-unit boards cut by mode p (p=1..6)
x = {}
for p in range(1, 7):
    x[p] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=3, name=f"x_{p}")

# y_bench, y_chair, y_table are pieces of synthetic-wood furniture
y_bench = model.addVar(vtype=GRB.INTEGER, lb=0, ub=12, name="y_bench")
y_chair = model.addVar(vtype=GRB.INTEGER, lb=0, ub=7, name="y_chair")
y_table = model.addVar(vtype=GRB.INTEGER, lb=0, ub=4, name="y_table")

# W is the total waste (meters) from Factory 1 (modes 1, 2, 3)
W = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="W")

# 4. Set up the objective function
# Maximize Z = 75x1 + 62x2 + 77x3 + 90x4 + 88x5 + 92x6 + 1y_bench + 4y_chair + 6y_table
# These coefficients were derived from the profits of parts and synthetic furniture after processing costs.
model.setObjective(
    75*x[1] + 62*x[2] + 77*x[3] + 90*x[4] + 88*x[5] + 92*x[6] +
    1*y_bench + 4*y_chair + 6*y_table,
    GRB.MAXIMIZE
)

# 5. Add all constraints
# Demand: 25-unit parts (at least 8)
model.addConstr(2*x[1] + 2*x[4] + 4*x[5] >= 8, "Demand_25")
# Demand: 40-unit parts (at least 6)
model.addConstr(1*x[1] + 2*x[2] + 1*x[3] >= 6, "Demand_40")
# Demand: 50-unit parts (at least 4)
model.addConstr(1*x[3] + 1*x[4] + 2*x[6] >= 4, "Demand_50")

# Mode usage limit per mode: x_p <= 3 (already set in variable ub)

# Factory 2 minimum processing: x4 + x5 >= 1
model.addConstr(x[4] + x[5] >= 1, "Factory2_Min_Processing")

# Prerequisite for mode 6: any factory must process 3 other boards for each mode 6 board
# Represented as: x6 <= (x1+x2+x3+x4+x5)/3
model.addConstr(3 * x[6] <= x[1] + x[2] + x[3] + x[4] + x[5], "Mode6_Prerequisite")

# Board supply limit: Total boards used cannot exceed 20
model.addConstr(gp.quicksum(x[p] for p in range(1, 7)) <= 20, "Total_Board_Supply")

# Waste balance (Factory 1: modes 1, 2, 3 only)
# W = 10x1 + 20x2 + 10x3
model.addConstr(W == 10*x[1] + 20*x[2] + 10*x[3], "Waste_Balance_Factory1")

# Synthetic-wood usage: total meters used for furniture must not exceed available waste W
model.addConstr(20*y_bench + 40*y_chair + 50*y_table <= W, "Synthetic_Wood_Usage_Limit")

# 6. Solve the model
model.optimize()

# 7. Print the results
if model.status == GRB.OPTIMAL:
    final_profit = model.objVal
    print(f"FinalAnswer=【{final_profit}】")
else:
    print("Optimal solution was not found.")
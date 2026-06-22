import gurobipy as gp
from gurobipy import GRB

# 1. Define Parameters
# Note: Units must be consistent. 
# Lengths in mm, Areas in mm^2, Forces in N, Stresses in MPa (N/mm^2), Moments in N*mm.
M_req_kNm = 200.0
M_req = M_req_kNm * 1e6  # Convert kN*m to N*mm

f_c_prime = 30.0   # MPa
f_y = 400.0        # MPa
phi = 0.9          # Resistance coefficient
rho_min = 0.035    # Minimum reinforcement ratio
k = 0.85           # Concrete strength correction factor
cost_area_coeff = 1.0       # Coefficient for concrete area in cost
cost_steel_ratio = 15.0     # Coefficient for steel area relative to concrete

# 2. Initialize Model
model = gp.Model("ReinforcedConcreteBeamDesign")

# Set NonConvex parameter to 2 to handle quadratic equality constraints 
# (e.g., area calculation b*h, moment calculation As*z)
model.Params.NonConvex = 2

# 3. Create Decision Variables
# Dimensions b and h
b = model.addVar(lb=200, ub=1000, vtype=GRB.CONTINUOUS, name="b")
h = model.addVar(lb=200, ub=1000, vtype=GRB.CONTINUOUS, name="h")

# Steel area As
A_s = model.addVar(lb=0, ub=20000, vtype=GRB.CONTINUOUS, name="A_s")

# 4. Create Auxiliary Variables
# Ac = Concrete Area = b * h
Ac = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Ac")

# a = Depth of equivalent rectangular stress block
# Equilibrium requires A_s * f_y = k * f_c' * b * a, so a is determined by variables
a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")

# z = Internal lever arm
z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")

# Mn = Nominal bending capacity
Mn = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Mn")

# 5. Set Objective Function
# Minimize Cost = (Concrete Area) + 15 * (Steel Area)
model.setObjective(cost_area_coeff * Ac + cost_steel_ratio * A_s, GRB.MINIMIZE)

# 6. Add Constraints

# Constraint: Concrete Area Definition
# Ac = b * h
model.addConstr(Ac == b * h, name="Calc_Ac")

# Constraint: Force Equilibrium
# Tensile force (As * fy) = Compressive force (k * fc' * b * a)
# Rearranged as constraint equation
model.addConstr(A_s * f_y == k * f_c_prime * b * a, name="Force_Equilibrium")

# Constraint: Internal Lever Arm Definition
# z = h - a/2
model.addConstr(z == h - 0.5 * a, name="Calc_z")

# Constraint: Nominal Moment Capacity Definition
# Mn = phi * As * fy * z
# Since As and z are variables, this is a quadratic constraint
model.addConstr(Mn == phi * f_y * A_s * z, name="Calc_Mn")

# Constraint: Bending Bearing Capacity Requirement
# Mn >= M_req
model.addConstr(Mn >= M_req, name="Moment_Capacity_Req")

# Constraint: Minimum Reinforcement Ratio
# rho = As / (b * h) >= rho_min
# To avoid division by variables, use As >= rho_min * (b * h)
# Substitute Ac = b * h
model.addConstr(A_s >= rho_min * Ac, name="Min_Reinforcement_Ratio")

# 7. Solve and Print Results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    print(f"Width b: {b.X:.2f} mm")
    print(f"Height h: {h.X:.2f} mm")
    print(f"Steel Area As: {A_s.X:.2f} mm^2")
    print(f"Stress Block a: {a.X:.2f} mm")
    print(f"Lever Arm z: {z.X:.2f} mm")
    print(f"Moment Capacity Mn: {Mn.X:.2e} N*mm")
    print(f"Objective Cost: {model.ObjVal:.4f}")
    
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")
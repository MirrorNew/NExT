import gurobipy as gp
import math

# 1. Define all parameter matrices and data inputs
G = 11500000.0         # Shear modulus (psi)
rho = 0.284            # Material density (lb/in³)
F_work = 300           # Working load (lbf)
tau_allowable = 80000  # Allowable shear stress (psi)
f_max = 4              # Maximum natural frequency (Hz)
delta_allowable = 0.5  # Minimum allowable compression (inch)
C_min, C_max = 4, 12   # Winding ratio bounds
outer_diameter_max = 1.5  # Maximum outer diameter (inch)
d_min, d_max = 0.1, 0.5   # Wire diameter bounds (inch)
D_min, D_max = 0.5, 6.0   # Spring center diameter bounds (inch)

# 2. Create model
model = gp.Model("SpringDesign")

# 3. Create decision variables
d = model.addVar(lb=d_min, ub=d_max, name="d")      # Wire diameter
D = model.addVar(lb=D_min, ub=D_max, name="D")      # Spring center diameter
N = model.addVar(lb=1, vtype=gp.GRB.INTEGER, name="N")  # Total number of coils

# 4. Create auxiliary substitution variables
C = model.addVar(lb=C_min, ub=C_max, name="C")      # Winding ratio C = D/d
K_w = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="K_w")  # Wahl correction factor
tau = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="tau")  # Shear stress
k = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="k")      # Spring stiffness
m = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="m")      # Spring mass
ratio_km = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="ratio_km")  # k/m ratio
sqrt_km = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="sqrt_km")    # sqrt(k/m)
f = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="f")      # Natural frequency
delta_max = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="delta_max")  # Max compression

# 5. Set up the objective function
# W = (π²/4) * d² * D * (N+2)
# We'll create an auxiliary variable for weight
weight = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="weight")
model.addConstr(weight == (math.pi**2 / 4) * d * d * D * (N + 2))
model.setObjective(weight, gp.GRB.MINIMIZE)

# 6. Add all constraints
# Allow non-convex constraints
model.Params.NonConvex = 2

# Constraint 1: Winding ratio definition C = D/d
# Rewrite as: C * d = D
model.addConstr(C * d == D, name="C_def")

# Constraint 2: Wahl correction factor K_w = (4C-1)/(4C-4) + 0.615/C
# Rewrite as: K_w*(4C-4)*C = C*(4C-1) + 0.615*(4C-4)
model.addConstr(K_w * (4*C - 4) * C == C * (4*C - 1) + 0.615 * (4*C - 4), name="K_w_def")

# Constraint 3: Shear stress τ = (8*F_work*D)/(π*d³) * K_w
# Rewrite as: τ*π*d³ = 8*F_work*D*K_w
model.addConstr(tau * math.pi * d * d * d == 8 * F_work * D * K_w, name="tau_def")

# Constraint 4: Shear stress limit
model.addConstr(tau <= tau_allowable, name="tau_limit")

# Constraint 5: Spring stiffness k = G*d⁴/(8*D³*N)
# Rewrite as: k*8*D³*N = G*d⁴
model.addConstr(k * 8 * D * D * D * N == G * d * d * d * d, name="k_def")

# Constraint 6: Spring mass m = ρ*π²*d²*D*(N+2)/4
# Rewrite as: 4*m = ρ*π²*d²*D*(N+2)
model.addConstr(4 * m == rho * math.pi**2 * d * d * D * (N + 2), name="m_def")

# Constraint 7: Ratio k/m
model.addConstr(ratio_km * m == k, name="ratio_km_def")

# Constraint 8: Square root of ratio_km
model.addGenConstrPow(ratio_km, sqrt_km, 0.5, name="sqrt_km_def")

# Constraint 9: Natural frequency f = (1/(2π)) * sqrt(k/m)
# Rewrite as: 2π*f = sqrt_km
model.addConstr(2 * math.pi * f == sqrt_km, name="f_def")

# Constraint 10: Frequency limit - "Natural frequency must be less than 4 Hz"
model.addConstr(f <= f_max, name="f_limit")

# Constraint 11: Maximum compression δ_max = 8*F_work*D³*N/(G*d⁴)
# Rewrite as: δ_max*G*d⁴ = 8*F_work*D³*N
model.addConstr(delta_max * G * d * d * d * d == 8 * F_work * D * D * D * N, name="delta_max_def")

# Constraint 12: Compression limit - "compression must not be less than 0.5"
model.addConstr(delta_max >= delta_allowable, name="delta_limit")

# Constraint 13: Outer diameter constraint
model.addConstr(D + d <= outer_diameter_max, name="outer_diameter_limit")

# 7. Solve the model and print results
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal solution found!")
    print(f"Wire diameter d = {d.X:.4f} inch")
    print(f"Spring center diameter D = {D.X:.4f} inch")
    print(f"Total number of coils N = {int(N.X)}")
    print(f"Winding ratio C = D/d = {C.X:.4f}")
    print(f"Spring weight W = {weight.X:.4f} lb")
    print(f"Natural frequency f = {f.X:.4f} Hz")
    print(f"Maximum compression δ_max = {delta_max.X:.4f} inch")
    print(f"Shear stress τ = {tau.X:.2f} psi (allowable: {tau_allowable} psi)")
    print(f"Outer diameter D+d = {D.X + d.X:.4f} inch (max: {outer_diameter_max})")
    print(f"FinalAnswer=【{weight.X:.4f}】")
else:
    print(f"Solution status: {model.status}")
    if model.status == gp.GRB.INFEASIBLE:
        model.computeIIS()
        print("IIS constraints:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"  {c.ConstrName}")
    print(f"FinalAnswer=【No optimal solution found】")
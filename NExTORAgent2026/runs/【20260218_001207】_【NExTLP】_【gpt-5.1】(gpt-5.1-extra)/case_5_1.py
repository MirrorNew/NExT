import gurobipy as gp
from gurobipy import GRB

# ======================
# 1. Define parameters
# ======================

Total_Portfolio_Value = 1000000
Total_Investment_Upper_Bound = 1000000
Pairwise_Investment_Sum_Upper_Bound = 700000
Liquidity_Min_Bond_and_CD = 200000
CD_Max_Investment = 300000
Real_Estate_Min_Proportion = 0.3
Stock_Max_Investment = 400000
Bond_Min_Investment = 100000
Diversification_Threshold = 500000
ESG_Min_Weighted_Average = 0.7
Risk_Factor_Max_Weighted_Average = 0.2
Additional_Risk_Reserve = 200000

Asset_Types = ['Stock S', 'Real Estate R', 'Bond B', 'Certificate of Deposit C', 'Hedging Product D']
Expected_Annual_Return = [0.06, 0.07, 0.05, 0.04, 0.1]
Minimum_Investment_USD = [100000, 300000, 100000, 100000, 500000]
Maximum_Investment_USD = [400000, 1000000, 1000000, 300000, 1000000]
ESG_Score = [0.5, 0.7, 0.8, 0.9, 0.3]
Risk_Factor = [0.3, 0.25, 0.1, 0.05, 0.8]

# Indices for convenience
idx_S = 0
idx_R = 1
idx_B = 2
idx_C = 3
idx_D = 4  # not used in the mathematical model, but kept for consistency

# A safe Big-M for indicator constraints based on total portfolio
M_big = Total_Portfolio_Value

# ======================
# 2. Create model
# ======================

model = gp.Model("XYZ_Asset_Allocation")

# ======================
# 3. Decision variables
# ======================

# Investment amounts (continuous)
S = model.addVar(lb=0.0, name="S")
R = model.addVar(lb=0.0, name="R")
B = model.addVar(lb=0.0, name="B")
C = model.addVar(lb=0.0, name="C")

# Reserve variable (continuous, for risk regulation; not part of invested 1M)
Reserve = model.addVar(lb=0.0, name="Reserve")

# Binary variables for diversification
y_S = model.addVar(vtype=GRB.BINARY, name="y_S")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_C = model.addVar(vtype=GRB.BINARY, name="y_C")

# Binary variable for risk reserve trigger
z_risk = model.addVar(vtype=GRB.BINARY, name="z_risk")

model.update()

# ======================
# 4. Objective function
# ======================

# Maximize annual return of invested 1M (hedging product D is not used here)
model.setObjective(
    Expected_Annual_Return[idx_S] * S +
    Expected_Annual_Return[idx_R] * R +
    Expected_Annual_Return[idx_B] * B +
    Expected_Annual_Return[idx_C] * C,
    GRB.MAXIMIZE
)

# ======================
# 5. Constraints
# ======================

# (1) Total investment equality and upper bound
model.addConstr(S + R + B + C == Total_Portfolio_Value, name="Total_Investment_Eq")
model.addConstr(S + R + B + C <= Total_Investment_Upper_Bound, name="Total_Investment_UB")

# (2) Pairwise caps
model.addConstr(S + R <= Pairwise_Investment_Sum_Upper_Bound, name="Pair_S_R")
model.addConstr(S + B <= Pairwise_Investment_Sum_Upper_Bound, name="Pair_S_B")
model.addConstr(S + C <= Pairwise_Investment_Sum_Upper_Bound, name="Pair_S_C")
model.addConstr(R + B <= Pairwise_Investment_Sum_Upper_Bound, name="Pair_R_B")
model.addConstr(R + C <= Pairwise_Investment_Sum_Upper_Bound, name="Pair_R_C")
model.addConstr(B + C <= Pairwise_Investment_Sum_Upper_Bound, name="Pair_B_C")

# (3) Liquidity: B + C >= 200000
model.addConstr(B + C >= Liquidity_Min_Bond_and_CD, name="Liquidity_B_C")

# (4) Individual bounds (minimum & maximum investments)

# Stock S
model.addConstr(S >= Minimum_Investment_USD[idx_S], name="S_min")
model.addConstr(S <= Maximum_Investment_USD[idx_S], name="S_max")

# Real Estate R
model.addConstr(R >= Minimum_Investment_USD[idx_R], name="R_min")
model.addConstr(R <= Maximum_Investment_USD[idx_R], name="R_max")

# Bond B
model.addConstr(B >= Minimum_Investment_USD[idx_B], name="B_min")
model.addConstr(B <= Maximum_Investment_USD[idx_B], name="B_max")

# Certificate of Deposit C
model.addConstr(C >= Minimum_Investment_USD[idx_C], name="C_min")
model.addConstr(C <= CD_Max_Investment, name="C_max")

# (5) Real estate minimum share: R >= 0.3 (S + R + B + C)
model.addConstr(
    R >= Real_Estate_Min_Proportion * (S + R + B + C),
    name="RealEstate_Min_Proportion"
)

# (6) Stock max investment (risk cap)
model.addConstr(S <= Stock_Max_Investment, name="Stock_Max_Investment")

# (7) Bond minimum investment (stability)
model.addConstr(B >= Bond_Min_Investment, name="Bond_Min_Investment")

# (8) Stock vs (R + B): S <= 0.5 (R + B)
model.addConstr(S <= 0.5 * (R + B), name="Stock_vs_RB")

# (9) Diversification: at least one asset >= Diversification_Threshold
#    Using indicator constraints instead of explicit Big-M linearization.

# If y_S = 1  ->  S >= Diversification_Threshold
model.addGenConstrIndicator(y_S, 1, S >= Diversification_Threshold, name="Ind_yS_1")
# If y_S = 0  ->  S <= Diversification_Threshold - tiny (use exact threshold as safe upper)
model.addGenConstrIndicator(y_S, 0, S <= Diversification_Threshold, name="Ind_yS_0")

model.addGenConstrIndicator(y_R, 1, R >= Diversification_Threshold, name="Ind_yR_1")
model.addGenConstrIndicator(y_R, 0, R <= Diversification_Threshold, name="Ind_yR_0")

model.addGenConstrIndicator(y_B, 1, B >= Diversification_Threshold, name="Ind_yB_1")
model.addGenConstrIndicator(y_B, 0, B <= Diversification_Threshold, name="Ind_yB_0")

model.addGenConstrIndicator(y_C, 1, C >= Diversification_Threshold, name="Ind_yC_1")
model.addGenConstrIndicator(y_C, 0, C <= Diversification_Threshold, name="Ind_yC_0")

# At least one asset category must be >= threshold
model.addConstr(y_S + y_R + y_B + y_C >= 1, name="Diversification_AtLeastOne")

# (10) ESG constraint:
# (0.5S + 0.7R + 0.8B + 0.9C) / (S + R + B + C) >= 0.7
# With total S+R+B+C = Total_Portfolio_Value, equivalently:
# 0.5S + 0.7R + 0.8B + 0.9C >= 0.7(S + R + B + C)
model.addConstr(
    ESG_Score[idx_S] * S +
    ESG_Score[idx_R] * R +
    ESG_Score[idx_B] * B +
    ESG_Score[idx_C] * C
    >= ESG_Min_Weighted_Average * (S + R + B + C),
    name="ESG_Constraint"
)

# (11) Risk constraint with conditional reserve:
# Average risk factor:
# (0.30S + 0.25R + 0.10B + 0.05C) / (S + R + B + C) <= 0.2 if no reserve;
# If higher, trigger additional reserve of 200000.
#
# Implement via indicator + Big-M relaxation:
# risk_sum <= 0.2 * total + M * z_risk
risk_sum = (Risk_Factor[idx_S] * S +
            Risk_Factor[idx_R] * R +
            Risk_Factor[idx_B] * B +
            Risk_Factor[idx_C] * C)

total_invest = S + R + B + C

# Indicator: if z_risk == 0, enforce risk_sum <= 0.2 * total_invest
model.addGenConstrIndicator(
    z_risk, 0,
    risk_sum <= Risk_Factor_Max_Weighted_Average * total_invest,
    name="Risk_Low_Indicator"
)

# Indicator: if z_risk == 1, allow risk above threshold (no upper restriction via this)
# Use a very loose upper bound so that constraint is non-binding when z_risk = 1
model.addGenConstrIndicator(
    z_risk, 1,
    risk_sum <= Risk_Factor_Max_Weighted_Average * total_invest + M_big,
    name="Risk_High_Indicator"
)

# Reserve is 200000 if z_risk == 1, 0 if z_risk == 0
model.addGenConstrIndicator(
    z_risk, 1,
    Reserve == Additional_Risk_Reserve,
    name="Reserve_On"
)
model.addGenConstrIndicator(
    z_risk, 0,
    Reserve == 0,
    name="Reserve_Off"
)

# Nonnegativity is already enforced by variable lb>=0

# ======================
# 6. Optimize
# ======================

model.Params.OutputFlag = 0  # turn off solver output for cleanliness; remove if needed
model.optimize()

# ======================
# 7. Print results
# ======================

if model.Status == GRB.OPTIMAL:
    S_val = S.X
    R_val = R.X
    B_val = B.X
    C_val = C.X
    Reserve_val = Reserve.X
    obj_val = model.ObjVal

    print(f"S (Stock) investment: {S_val:.2f}")
    print(f"R (Real Estate) investment: {R_val:.2f}")
    print(f"B (Bond) investment: {B_val:.2f}")
    print(f"C (Certificate of Deposit) investment: {C_val:.2f}")
    print(f"Reserve (if any): {Reserve_val:.2f}")
    print(f"Maximum annualized return: {obj_val:.2f}")
else:
    print("No optimal solution found.")
    obj_val = float('nan')

# As requested: print FinalAnswer as the maximum annualized return
print(f"FinalAnswer=【{obj_val}】")
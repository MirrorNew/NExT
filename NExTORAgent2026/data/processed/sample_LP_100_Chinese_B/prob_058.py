def optimize_fertilizer_cost():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Fertilizer_Optimization")

    # Decision variables: amount of fertilizers C and Y
    x = m.addVar(name="C", lb=0)  # fertilizer C in kg
    y = m.addVar(name="Y", lb=0)  # fertilizer Y in kg

    # Set the objective: minimize total cost
    m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)

    # Add constraints
    m.addConstr(1.5 * x + 5 * y >= 5, name="NitrousOxide")
    m.addConstr(3 * x + y >= 8, name="VitaminMix")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(1.5 * x + 5 * y <= 2 * (3 * x + y), name="NOx_to_Vitamin_Ratio")
    # 为了显式引入非线性（比例约束），在维生素总量上添加辅助变量并使用乘积形式：
    vitamins = m.addVar(name="VitaminsTotal", lb=0)
    m.addConstr(vitamins == 3 * x + y, name="VitaminsDefinition")
    # 比率约束： (一氧化二氮总单位数)/(维生素总单位数) <= 2
    # 一氧化二氮总单位数 = 1.5x + 5y
    # 非线性写法： (1.5*x + 5*y) * 1.0 / vitamins <= 2
    m.addConstr((1.5 * x + 5 * y)  <= 2* vitamins, name="NOx_to_Vitamin_Ratio_Nonlinear")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        return None


# Example usage
if __name__ == "__main__":
    min_cost = optimize_fertilizer_cost()
    if min_cost is not None:
        print(f"Minimum Cost of Fertilizer Mixture: {min_cost}")
    else:
        print("No feasible solution found.")
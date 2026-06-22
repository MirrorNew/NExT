import gurobipy as gp

# 参数列表
calorie_min = 2000
calorie_max = 2500
protein_min = 50
vitamin_C_min = 100
fat_max = 70
vegetables_min = 100
chicken_max = 300
beans_max = 400
incentive_meal_days = 1
incentive_day_cal_min = 2500
incentive_day_cal_max = 3000

table_1_limits = {
    'minimum_weekly_purchase_quantity': {'Pack A': 3, 'Pack B': 1, 'Pack C': 1, 'Pack D': 5, 'Vegetables': None},
    'daily_delivery_limit': {'Pack A': 2, 'Pack B': 5, 'Pack C': 5, 'Pack D': 1, 'Vegetables': None}
}

# 包成分表（g/包）
pack_composition = {
    'A': {'Rice': 25, 'Chicken': 50, 'Beans': 0, 'Milk': 0},
    'B': {'Rice': 25, 'Chicken': 40, 'Beans': 0, 'Milk': 0},
    'C': {'Rice': 10, 'Chicken': 20, 'Beans': 20, 'Milk': 0},
    'D': {'Rice': 0, 'Chicken': 0, 'Beans': 0, 'Milk': 50}
}

# 营养成分和成本（每100g）
nutrient_cost = {
    'Rice': {'Calories': 360, 'Protein': 6, 'Fat': 1, 'Vitamin C': 0, 'Cost': 0.5},
    'Chicken': {'Calories': 200, 'Protein': 20, 'Fat': 15, 'Vitamin C': 0, 'Cost': 2.0},
    'Beans': {'Calories': 150, 'Protein': 8, 'Fat': 1, 'Vitamin C': 5, 'Cost': 1.0},
    'Milk': {'Calories': 60, 'Protein': 3, 'Fat': 4, 'Vitamin C': 5, 'Cost': 1.5},
    'Vegetables': {'Calories': 50, 'Protein': 2, 'Fat': 0, 'Vitamin C': 20, 'Cost': 1.0}
}

# 创建模型
model = gp.Model("Weekly_Meal_Plan_Optimization")

# 天数
days = 7
day_indices = range(1, days+1)

# 变量
# 每天每个包的数量（整数）
xA = model.addVars(day_indices, vtype=gp.GRB.INTEGER, name="xA", ub=2)  # Pack A
xB = model.addVars(day_indices, vtype=gp.GRB.INTEGER, name="xB", ub=5)  # Pack B
xC = model.addVars(day_indices, vtype=gp.GRB.INTEGER, name="xC", ub=5)  # Pack C
xD = model.addVars(day_indices, vtype=gp.GRB.INTEGER, name="xD", ub=1)  # Pack D

# 每天蔬菜量（连续）
v = model.addVars(day_indices, vtype=gp.GRB.CONTINUOUS, name="v", lb=100)

# 激励餐指示变量（二进制）
y = model.addVars(day_indices, vtype=gp.GRB.BINARY, name="y")

# 目标函数
cost_packA_per_unit = (pack_composition['A']['Rice']/100 * nutrient_cost['Rice']['Cost'] + 
                       pack_composition['A']['Chicken']/100 * nutrient_cost['Chicken']['Cost'])
cost_packB_per_unit = (pack_composition['B']['Rice']/100 * nutrient_cost['Rice']['Cost'] + 
                       pack_composition['B']['Chicken']/100 * nutrient_cost['Chicken']['Cost'])
cost_packC_per_unit = (pack_composition['C']['Rice']/100 * nutrient_cost['Rice']['Cost'] + 
                       pack_composition['C']['Chicken']/100 * nutrient_cost['Chicken']['Cost'] + 
                       pack_composition['C']['Beans']/100 * nutrient_cost['Beans']['Cost'])
cost_packD_per_unit = (pack_composition['D']['Milk']/100 * nutrient_cost['Milk']['Cost'])

objective_expr = gp.quicksum(cost_packA_per_unit * xA[j] + 
                            cost_packB_per_unit * xB[j] + 
                            cost_packC_per_unit * xC[j] + 
                            cost_packD_per_unit * xD[j] + 
                            (nutrient_cost['Vegetables']['Cost']/100) * v[j] 
                            for j in day_indices)
model.setObjective(objective_expr, gp.GRB.MINIMIZE)

# 约束
# 1. 恰好一个激励餐日
model.addConstr(gp.quicksum(y[j] for j in day_indices) == 1, name="exactly_one_incentive_day")

# 每日约束
for j in day_indices:
    # 计算每日总卡路里（每包成分转换为卡路里）
    cal_expr = (
        (pack_composition['A']['Rice']/100 * nutrient_cost['Rice']['Calories'] + 
         pack_composition['A']['Chicken']/100 * nutrient_cost['Chicken']['Calories']) * xA[j] +
        (pack_composition['B']['Rice']/100 * nutrient_cost['Rice']['Calories'] + 
         pack_composition['B']['Chicken']/100 * nutrient_cost['Chicken']['Calories']) * xB[j] +
        (pack_composition['C']['Rice']/100 * nutrient_cost['Rice']['Calories'] + 
         pack_composition['C']['Chicken']/100 * nutrient_cost['Chicken']['Calories'] + 
         pack_composition['C']['Beans']/100 * nutrient_cost['Beans']['Calories']) * xC[j] +
        (pack_composition['D']['Milk']/100 * nutrient_cost['Milk']['Calories']) * xD[j] +
        (nutrient_cost['Vegetables']['Calories']/100) * v[j]
    )
    
    # 2. 卡路里下限（2000 + 500*y_j）
    model.addConstr(cal_expr >= calorie_min + 500 * y[j], name=f"cal_min_day_{j}")
    
    # 3. 卡路里上限（2500 + 500*y_j）
    model.addConstr(cal_expr <= calorie_max + 500 * y[j], name=f"cal_max_day_{j}")
    
    # 4. 蛋白质需求
    protein_expr = (
        (pack_composition['A']['Rice']/100 * nutrient_cost['Rice']['Protein'] + 
         pack_composition['A']['Chicken']/100 * nutrient_cost['Chicken']['Protein']) * xA[j] +
        (pack_composition['B']['Rice']/100 * nutrient_cost['Rice']['Protein'] + 
         pack_composition['B']['Chicken']/100 * nutrient_cost['Chicken']['Protein']) * xB[j] +
        (pack_composition['C']['Rice']/100 * nutrient_cost['Rice']['Protein'] + 
         pack_composition['C']['Chicken']/100 * nutrient_cost['Chicken']['Protein'] + 
         pack_composition['C']['Beans']/100 * nutrient_cost['Beans']['Protein']) * xC[j] +
        (pack_composition['D']['Milk']/100 * nutrient_cost['Milk']['Protein']) * xD[j] +
        (nutrient_cost['Vegetables']['Protein']/100) * v[j]
    )
    model.addConstr(protein_expr >= protein_min, name=f"protein_min_day_{j}")
    
    # 5. 维生素C需求
    vitC_expr = (
        (pack_composition['C']['Beans']/100 * nutrient_cost['Beans']['Vitamin C']) * xC[j] +
        (pack_composition['D']['Milk']/100 * nutrient_cost['Milk']['Vitamin C']) * xD[j] +
        (nutrient_cost['Vegetables']['Vitamin C']/100) * v[j]
    )
    model.addConstr(vitC_expr >= vitamin_C_min, name=f"vitC_min_day_{j}")
    
    # 6. 脂肪限制
    fat_expr = (
        (pack_composition['A']['Rice']/100 * nutrient_cost['Rice']['Fat'] + 
         pack_composition['A']['Chicken']/100 * nutrient_cost['Chicken']['Fat']) * xA[j] +
        (pack_composition['B']['Rice']/100 * nutrient_cost['Rice']['Fat'] + 
         pack_composition['B']['Chicken']/100 * nutrient_cost['Chicken']['Fat']) * xB[j] +
        (pack_composition['C']['Rice']/100 * nutrient_cost['Rice']['Fat'] + 
         pack_composition['C']['Chicken']/100 * nutrient_cost['Chicken']['Fat'] + 
         pack_composition['C']['Beans']/100 * nutrient_cost['Beans']['Fat']) * xC[j] +
        (pack_composition['D']['Milk']/100 * nutrient_cost['Milk']['Fat']) * xD[j]
    )
    model.addConstr(fat_expr <= fat_max, name=f"fat_max_day_{j}")
    
    # 7. 蔬菜最小值已在变量定义中设置 lb=100
    
    # 8. 鸡肉最大值
    chicken_expr = (
        pack_composition['A']['Chicken'] * xA[j] + 
        pack_composition['B']['Chicken'] * xB[j] + 
        pack_composition['C']['Chicken'] * xC[j]
    )
    model.addConstr(chicken_expr <= chicken_max, name=f"chicken_max_day_{j}")
    
    # 9. 豆类最大值
    beans_expr = pack_composition['C']['Beans'] * xC[j]
    model.addConstr(beans_expr <= beans_max, name=f"beans_max_day_{j}")

# 10. 每周最小购买量
model.addConstr(gp.quicksum(xA[j] for j in day_indices) >= table_1_limits['minimum_weekly_purchase_quantity']['Pack A'], name="weekly_min_A")
model.addConstr(gp.quicksum(xB[j] for j in day_indices) >= table_1_limits['minimum_weekly_purchase_quantity']['Pack B'], name="weekly_min_B")
model.addConstr(gp.quicksum(xC[j] for j in day_indices) >= table_1_limits['minimum_weekly_purchase_quantity']['Pack C'], name="weekly_min_C")
model.addConstr(gp.quicksum(xD[j] for j in day_indices) >= table_1_limits['minimum_weekly_purchase_quantity']['Pack D'], name="weekly_min_D")

# 求解
model.optimize()

# 输出结果
if model.status == gp.GRB.OPTIMAL:
    total_cost = model.ObjVal
    print("Optimal solution found.")
    print(f"Total weekly cost: {total_cost:.2f} yuan")
    print("\nDaily plan:")
    for j in day_indices:
        print(f"Day {j}: A={xA[j].X:.0f}, B={xB[j].X:.0f}, C={xC[j].X:.0f}, D={xD[j].X:.0f}, Vegetables={v[j].X:.2f}g, Incentive={y[j].X:.0f}")
    
    # 计算总采购量
    total_A = sum(xA[j].X for j in day_indices)
    total_B = sum(xB[j].X for j in day_indices)
    total_C = sum(xC[j].X for j in day_indices)
    total_D = sum(xD[j].X for j in day_indices)
    total_veg = sum(v[j].X for j in day_indices)
    print("\nWeekly totals:")
    print(f"Pack A: {total_A:.0f} bags")
    print(f"Pack B: {total_B:.0f} bags")
    print(f"Pack C: {total_C:.0f} bags")
    print(f"Pack D: {total_D:.0f} bags")
    print(f"Vegetables: {total_veg:.2f}g")
    
    # 按照要求输出最终答案
    print(f"FinalAnswer=【{total_cost:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【No solution found】")
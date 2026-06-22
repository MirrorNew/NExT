from gurobipy import Model, GRB, quicksum


def solve_supermarket_location():
    # --- 模型初始化 ---
    model = Model("SupermarketSetCovering_WithFixedCost")

    # --- 集合定义 ---
    # 住宅区既是潜在的建店地点，也是需要被覆盖的区域
    locations = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    areas_to_cover = locations[:]  # 创建一个副本

    # --- 参数定义 ---
    # coverage_data[loc] 是一个列表，包含地点loc建店可以覆盖的所有区域
    coverage_data = {
        "A": ["A", "C", "E", "G", "H", "I"],
        "B": ["B", "H", "I"],
        "C": ["A", "C", "G", "H", "I"],
        "D": ["D", "J"],
        "E": ["A", "E", "G"],
        "F": ["F", "J", "K"],
        "G": ["A", "C", "E", "G"],
        "H": ["A", "B", "C", "H", "I"],
        "I": ["A", "B", "C", "H", "I"],
        "J": ["D", "F", "J", "K", "L"],
        "K": ["F", "J", "K", "L"],
        "L": ["J", "K", "L"],
    }

    # --- 决策变量 ---
    # build[loc]: 是否在地点loc建店 (1=是, 0=否)
    build = model.addVars(locations, name="BuildStore", vtype=GRB.BINARY)

    # ❤ Non-linearity is introduced. ❤
    # 额外二进制变量：是否需要启用统一的信息管理与冷链配送系统（即店数是否超过 4 家）
    y = model.addVar(name="UseUnifiedSystem", vtype=GRB.BINARY)

    # --- 辅助变量：建店总数 ---
    total_build = model.addVar(name="TotalBuild", vtype=GRB.INTEGER)

    # 约束：总建店数等于各建店变量之和
    model.addConstr(total_build == quicksum(build[loc] for loc in locations),
                    name="TotalBuildDef")

    # 线性化“超过 4 家则产生 50000 元固定成本”的非线性条件
    # 当 total_build >= 5 时，应有 y = 1；当 total_build <= 4 时，应有 y = 0。
    # 使用 Big-M 技巧构造：
    M = len(locations)  # 最大可能门店数

    # 保证 total_build > 4 时 y 必须为 1：
    # total_build - 5 + 1 <= M * y  => total_build - 4 <= M * y
    model.addConstr(total_build - 4 <= M * y, name="Link_y_lower")

    # 保证 total_build <= 4 时 y 可以为 0 且不强迫 y=1：
    # total_build >= 5*y + 0  => 若 y=1 则 total_build >= 5（超过4）；若 y=0 则无约束
    model.addConstr(total_build >= 5 * y, name="Link_y_upper")

    # --- 目标函数 ---
    # ❤ Non-linearity is introduced. ❤
    # 原来：最小化建店数量
    # model.setObjective(quicksum(build[loc] for loc in locations), GRB.MINIMIZE)
    # 现在：最小化“门店数量 + 额外固定成本 50000（若店数>4）”
    extra_fixed_cost = 50000
    model.setObjective(
        quicksum(build[loc] for loc in locations) + extra_fixed_cost * y,
        GRB.MINIMIZE
    )

    # --- 约束条件: 每个区域至少被一个店覆盖 ---
    for area in areas_to_cover:
        # 找出哪些潜在店址可以覆盖当前区域 area
        covering_locations = []
        for loc_candidate in locations:
            if area in coverage_data[loc_candidate]:
                covering_locations.append(loc_candidate)

        # 添加约束: sum(build[loc] for loc in covering_locations) >= 1
        if covering_locations:  # 确保列表不为空
            model.addConstr(quicksum(build[loc] for loc in covering_locations)
                            >= 1,
                            name=f"CoverArea_{area}")
        else:
            print(f"警告: 区域 {area} 无法被任何潜在店址覆盖。请检查数据。")

    # --- 模型求解 ---
    model.optimize()

    # --- 打印结果 ---
    if model.status == GRB.OPTIMAL:
        print(f"\n找到最优选址方案!")
        print(f"目标值（门店数量 + 额外固定成本项）: {model.objVal:.0f}")

        actual_store_count = sum(1 for loc in locations if build[loc].X > 0.5)
        print(f"最终建店数量: {actual_store_count}")
        print(f"是否需要额外统一系统 (店数>4): {'是' if y.X > 0.5 else '否'}")
        if y.X > 0.5:
            print(f"产生一次性额外固定成本: {extra_fixed_cost} 元")
        else:
            print("不产生一次性额外固定成本。")

        print("\n建店地点:")
        for loc in locations:
            if build[loc].X > 0.5:  # 检查二元变量是否为1
                print(f"  - 在区域 {loc} 建店")

        print("\n各区域覆盖情况:")
        for area in areas_to_cover:
            covered_by_stores = []
            for loc in locations:
                if build[loc].X > 0.5 and area in coverage_data[loc]:
                    covered_by_stores.append(loc)
            print(
                f"  区域 {area} 被以下店址覆盖: {', '.join(covered_by_stores) if covered_by_stores else '未被覆盖 (错误!)'}"
            )

    elif model.status == GRB.INFEASIBLE:
        print("模型不可行。请检查约束条件或覆盖数据。")
        print("可能原因：某个区域无法被任何潜在店址覆盖。")
    elif model.status == GRB.UNBOUNDED:
        print("模型无界。(在此问题中不应发生)")
    else:
        print(f"优化过程因状态码 {model.status} 而停止。")


if __name__ == '__main__':
    solve_supermarket_location()
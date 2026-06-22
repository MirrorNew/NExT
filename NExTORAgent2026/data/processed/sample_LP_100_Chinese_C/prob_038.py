import gurobipy as gp
from gurobipy import GRB


def solve_warehouse_optimization(
    products=[1, 2, 3, 4, 5, 6, 7, 8],
    volume={1: 80, 2: 60, 3: 40, 4: 70, 5: 50, 6: 30, 7: 90, 8: 20},
    profit={
        1: 1200,
        2: 900,
        3: 700,
        4: 1100,
        5: 800,
        6: 500,
        7: 1300,
        8: 300
    },
    warehouse_capacity=500,
    incompatible_pairs=[(1, 3), (1, 5), (2, 4), (3, 1), (3, 6), (4, 2),
                        (4, 7), (5, 1), (6, 3), (7, 4)],
    volume_threshold=400,
    extra_fixed_cost=2000
):
    """
    Solves the warehouse storage optimization problem with
    an additional non-linear-like cost rule:
    If total volume used > volume_threshold, incur extra_fixed_cost once.
    """

    model = gp.Model("Warehouse_Storage_Optimization")

    # Decision Variables
    x = model.addVars(products, vtype=GRB.BINARY, name="select")

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # model.setObjective(gp.quicksum(profit[p] * x[p] for p in products),
    #                    GRB.MAXIMIZE)

    # Auxiliary binary variable indicating whether extra storage is triggered
    y = model.addVar(vtype=GRB.BINARY, name="use_extra_storage")

    # Total volume expression
    total_volume = gp.quicksum(volume[p] * x[p] for p in products)

    # Big-M linearization: if total_volume > volume_threshold then y must be 1
    # Choose M as the warehouse capacity (an upper bound on total volume)
    M = warehouse_capacity

    # total_volume - volume_threshold <= M * y
    # When y = 0  → total_volume <= volume_threshold
    # When y = 1  → constraint becomes non-binding (allows > threshold)
    model.addConstr(total_volume - volume_threshold <= M * y,
                    name="ExtraStorageTrigger")

    # New objective: maximize net profit (product profit minus possible fixed cost)
    model.setObjective(
        gp.quicksum(profit[p] * x[p] for p in products) - extra_fixed_cost * y,
        GRB.MAXIMIZE
    )

    # Capacity constraint
    model.addConstr(
        total_volume <= warehouse_capacity,
        "Capacity"
    )

    # Incompatibility constraints
    for (p, q) in incompatible_pairs:
        model.addConstr(x[p] + x[q] <= 1, f"Incompatible_{p}_{q}")

    # Solve the Model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        selected_products = [p for p in products if x[p].X > 0.5]
        total_profit = sum(profit[p] for p in selected_products)
        used_volume = sum(volume[p] for p in selected_products)
        extra_storage_used = int(round(y.X))
        net_profit = model.ObjVal
        return {
            "status": "optimal",
            "obj": net_profit,
            "selected_products": selected_products,
            "total_profit_before_cost": total_profit,
            "used_volume": used_volume,
            "extra_storage_used": extra_storage_used
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_warehouse_optimization()
    print(result)
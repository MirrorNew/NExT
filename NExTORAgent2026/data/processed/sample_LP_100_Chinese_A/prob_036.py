import gurobipy as gp
from gurobipy import GRB
import math


def solve_bakery_distribution():
    BakerySupply=[200, 150, 250, 180]
    StoreDemand=[120, 100, 130, 90, 80]
    ShippingDistance=[[10, 15, 20, 25, 30],
                      [12, 8, 18, 22, 28],
                      [14, 10, 16, 20, 26],
                      [16, 12, 14, 18, 24]]
    """
    Solves the bakery distribution problem with non-linear (exponential) shipping costs:
    For a route with shipped quantity x, total cost is:
        sqrt(distance) * (1.3 ** x) * x
    """

    # --- 1. Model Creation ---
    model = gp.Model("Bakery Distribution Optimization - Nonlinear")

    # --- 2. Parameters & Sets ---
    BakeryNum = len(BakerySupply)
    StoreNum = len(StoreDemand)
    Bakeries = range(BakeryNum)
    Stores = range(StoreNum)

    # Distance-based base cost: sqrt(distance)
    BaseShippingCost = [[math.sqrt(dist) for dist in row]
                        for row in ShippingDistance]

    # --- 3. Decision Variables ---
    # Integer number of loaves shipped from bakery b to store s
    ShipAmount = model.addVars(Bakeries,
                               Stores,
                               vtype=GRB.INTEGER,
                               name="ShipAmount")

    # --- 4. Objective Function ---
    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(
    #     gp.quicksum(ShippingCost[b][s] * ShipAmount[b, s]
    #                 for b in Bakeries for s in Stores),
    #     GRB.MINIMIZE)

    # Nonlinear objective:
    # For each (b, s), cost = sqrt(distance) * 1.3^x * x, where x = ShipAmount[b,s]

    Y = model.addVars(BakeryNum, StoreNum)
    # 假设 Y == 1.3 ** ShipAmount[b, s]
    LogY= model.addVars(BakeryNum, StoreNum)

    for b in Bakeries:
        for s in Stores:
            model.addGenConstrLog(Y[b, s], LogY[b, s])
            model.addConstr(LogY[b, s] == ShipAmount[b, s] * math.log(1.1) )

    model.setObjective(
        gp.quicksum(
            BaseShippingCost[b][s] *
            (Y[b, s]) *
            ShipAmount[b, s]
            for b in Bakeries for s in Stores
        ),
        GRB.MINIMIZE
    )

    # --- 5. Constraints ---
    # Constraint 1: Bakery supply constraint
    model.addConstrs(
        (gp.quicksum(ShipAmount[b, s] for s in Stores) <= BakerySupply[b]
         for b in Bakeries),
        name="BakerySupply")

    # Constraint 2: Store demand constraint
    model.addConstrs(
        (gp.quicksum(ShipAmount[b, s] for b in Bakeries) == StoreDemand[s]
         for s in Stores),
        name="StoreDemand")

    # --- 6. Solve the Model ---
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "solution": {(b, s): ShipAmount[b, s].X
                         for b in Bakeries for s in Stores}
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_bakery_distribution()
    print(result)
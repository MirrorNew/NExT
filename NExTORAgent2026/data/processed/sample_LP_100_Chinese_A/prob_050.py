import gurobipy as gp
from gurobipy import GRB


def solve_fuel_production_optimization(
):
    Inventory = [8000, 12000]
    QualityRating = [12, 6]
    Revenue = [30, 25]
    DemandperMarketingDollar = [8, 15]
    MinQualityRating = [10, 8]
    """
    Models and solves the fuel production and marketing optimization problem
    with a nonlinear demand function for bioethanol (fuel 0):
        demand_0 = (8 * MarketingBudget_0)^1.2

    Notes:
    - This uses Gurobi's general function mechanism to represent the
      nonlinear concave function y = (8x)^1.2.
    """

    # Create a new model
    model = gp.Model("Fuel Production and Marketing Optimization (Nonlinear Demand)")

    # Sets
    Fuels = range(len(Revenue))
    Feedstocks = range(len(Inventory))

    # Decision Variables
    # Amount of fuel f produced from feedstock s
    ProductionAmount = {}
    for f in Fuels:
        for s in Feedstocks:
            ProductionAmount[f, s] = model.addVar(
                vtype=GRB.CONTINUOUS,
                name=f"ProductionAmount_{f+1}_{s+1}"
            )

    # Marketing budget for each fuel
    MarketingBudget = model.addVars(
        Fuels,
        vtype=GRB.CONTINUOUS,
        name="MarketingBudget"
    )

    # Auxiliary variable: realized demand for each fuel
    Demand = model.addVars(
        Fuels,
        vtype=GRB.CONTINUOUS,
        name="Demand"
    )

    # Define nonlinear demand function for bioethanol (fuel 0):
    # demand_0 = (8 * MarketingBudget_0)^1.2
    def ethanol_demand_func(x):
        return (8.0 * x) ** 1.2


    Y = model.addVar()
    model.addConstr(Y == 8.0 * MarketingBudget[0])
    model.addGenConstrPow(Y, Demand[0], 1.2)
    # 错误❌️
    # model.addGenConstrUser(
    #     MarketingBudget[0],
    #     Demand[0],
    #     ethanol_demand_func,
    #     name="EthanolDemandFunc"
    # )

    # For biodiesel (fuel 1), demand remains linear: demand_1 = 15 * MarketingBudget_1
    model.addConstr(
        Demand[1] == DemandperMarketingDollar[1] * MarketingBudget[1],
        name="BiodieselDemandLinear"
    )

    # Objective: Maximize profit (revenue - marketing costs)
    # revenue = price * quantity sold; quantity sold = total production of each fuel
    obj = gp.quicksum(
        Revenue[f] * gp.quicksum(ProductionAmount[f, s] for s in Feedstocks)
        for f in Fuels
    ) - gp.quicksum(MarketingBudget[f] for f in Fuels)

    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint 1: Inventory constraint
    for s in Feedstocks:
        model.addConstr(
            gp.quicksum(ProductionAmount[f, s] for f in Fuels) <= Inventory[s],
            f"Inventory_{s+1}"
        )

    # Constraint 2: Quality rating constraint
    for f in Fuels:
        model.addConstr(
            gp.quicksum(QualityRating[s] * ProductionAmount[f, s] for s in Feedstocks) >=
            MinQualityRating[f] * gp.quicksum(ProductionAmount[f, s] for s in Feedstocks),
            f"Quality_{f+1}"
        )

    # Constraint 3: Demand constraints
    for f in Fuels:
        # ❤ Non-linearity is introduced. ❤
        # model.addConstr(
        #     DemandperMarketingDollar[f] * MarketingBudget[f] >=
        #     gp.quicksum(ProductionAmount[f, s] for s in Feedstocks),
        #     f"Demand_{f+1}"
        # )
        model.addConstr(
            Demand[f] >= gp.quicksum(ProductionAmount[f, s] for s in Feedstocks),
            f"Demand_{f+1}"
        )

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "MarketingBudget": {f: MarketingBudget[f].X for f in Fuels},
            "Demand": {f: Demand[f].X for f in Fuels}
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_fuel_production_optimization()
    print(result)
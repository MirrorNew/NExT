import gurobipy as gp
from gurobipy import GRB


def solve_package_shipping_optimization():
    AirFreightCapacity=150
    GroundShippingCapacity=210
    AirFreightRequirement=[17, 5]    # [Standard, Express]
    GroundShippingRequirement=[30, 13] # [Standard, Express]
    RevenuePerPackage_Standard=40
    ExpressRevenueCoef=0.2
    """
    Models and solves the (now non-linear) package shipping optimization problem.

    Decision variables
    ------------------
    PackagesShipped[p, s] : integer
        Number of packages of type p shipped by method s
        p = 0: Standard, p = 1: Express
        s = 0: Air,       s = 1: Ground

    a : number of express packages shipped by air  = PackagesShipped[1, 0]
    b : number of express packages shipped by ground = PackagesShipped[1, 1]
    c : nonnegative continuous variable, "express operation intensity"

    Objective
    ---------
    Maximize:
        40 * (standard by air + standard by ground)
        + 0.02 * a * b * c

    Subject to capacity constraints for air and ground.
    """
    # Create a new model
    model = gp.Model("Package_Shipping_Optimization_Nonlinear")

    # Sets
    # 0: Standard, 1: Express
    P = range(2)
    # 0: Air freight, 1: Ground shipping
    S = range(2)

    # Decision Variables
    PackagesShipped = model.addVars(P, S, vtype=GRB.INTEGER, lb=0, name="PackagesShipped")

    # Additional decision variable: express operation intensity level c
    c = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,ub=1, name="ExpressIntensity")

    # Convenience aliases for express shipment quantities
    a = PackagesShipped[1, 0]  # Express via air
    b = PackagesShipped[1, 1]  # Express via ground

    # Objective Function: Maximize total revenue

    # ❤ Non-linearity is introduced. ❤
    # revenue = gp.quicksum(PackagesShipped[p, s] * RevenuePerPackage[p] for p in P for s in S)

    # Standard revenue: linear, only type 0 (Standard)
    standard_revenue = RevenuePerPackage_Standard * (
        PackagesShipped[0, 0] + PackagesShipped[0, 1]
    )

    # Express revenue: non-linear term 0.02 * a * b * c
    Y = model.addVar()
    model.addConstr(Y == a * b)
    model.addConstr(b >= 2)
    model.addConstr(a >= 1)
    express_revenue = ExpressRevenueCoef * Y * c

    revenue = standard_revenue + express_revenue
    model.setObjective(revenue, GRB.MAXIMIZE)

    # Constraints
    # 1. Air freight capacity constraint
    model.addConstr(
        gp.quicksum(PackagesShipped[p, 0] * AirFreightRequirement[p] for p in P)
        <= AirFreightCapacity,
        "AirFreightCapacity"
    )

    # 2. Ground shipping capacity constraint
    model.addConstr(
        gp.quicksum(PackagesShipped[p, 1] * GroundShippingRequirement[p] for p in P)
        <= GroundShippingCapacity,
        "GroundShippingCapacity"
    )

    # (Optional) you might want to bound c in practice, e.g., c <= 100,
    # but the problem statement does not specify an upper bound, so we omit it.

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "PackagesShipped": {
                (p, s): PackagesShipped[p, s].X for p in P for s in S
            },
            "ExpressIntensity": c.X,
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_package_shipping_optimization()
    print(result)
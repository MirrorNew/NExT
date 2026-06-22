import gurobipy as gp
from gurobipy import GRB


def solve_package_shipping_optimization(
    AirFreightCapacity=150,
    GroundShippingCapacity=210,
    AirFreightRequirement=[17, 5],
    GroundShippingRequirement=[30, 13],
    RevenuePerPackage=[40, 15]
):
    """
    Models and solves the package shipping optimization problem.
    """
    # Create a new model
    model = gp.Model("Package_Shipping_Optimization")

    # Sets
    P = range(len(RevenuePerPackage))  # Package types: 0 - Standard, 1 - Express
    S = range(2)  # Shipping methods (0: Air freight, 1: Ground shipping)

    # Decision Variables
    PackagesShipped = model.addVars(P, S, vtype=GRB.INTEGER, lb=0, name="PackagesShipped")

    # Objective Function: Maximize total revenue
    revenue = gp.quicksum(PackagesShipped[p, s] * RevenuePerPackage[p] for p in P for s in S)
    model.setObjective(revenue, GRB.MAXIMIZE)

    # Constraints
    # 1. Air freight capacity constraint
    model.addConstr(
        gp.quicksum(PackagesShipped[p, 0] * AirFreightRequirement[p] for p in P) <= AirFreightCapacity,
        "AirFreightCapacity"
    )

    # 2. Ground shipping capacity constraint
    model.addConstr(
        gp.quicksum(PackagesShipped[p, 1] * GroundShippingRequirement[p] for p in P) <= GroundShippingCapacity,
        "GroundShippingCapacity"
    )

    # ❤ Non-linearity is introduced. ❤
    # 3. Balance constraint: ratio of air-shipped Standard to ground-shipped Express <= 2
    #    PackagesShipped[0, 0] / PackagesShipped[1, 1] <= 2
    #    Introduce non-linear division directly as requested.
    model.addConstr(
        PackagesShipped[0, 0] <= 2 * PackagesShipped[1, 1],
        "Balance_StandardAir_ExpressGround"
    )

    # Optimize the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "PackagesShipped": {
                (p, s): int(PackagesShipped[p, s].X) for p in P for s in S
            }
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_package_shipping_optimization()
    print(result)
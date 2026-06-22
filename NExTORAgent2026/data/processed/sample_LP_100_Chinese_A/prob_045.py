import gurobipy as gp
from gurobipy import GRB
import math


def solve_supplier_shipment_optimization(
):
    CostPerShipment = [5.2, 4.7, 3.5]
    Percent = [
        [45, 35, 20],
        [30, 45, 25],
        [15, 20, 65]
    ]
    Demand = [500, 300, 300]
    MaxShipmentNominal = 700  # 名义单月运力上限
    months = 12
    # Create a new model
    model = gp.Model("Supplier Shipment Optimization with Seasonal Capacity")

    # Sets
    Suppliers = range(len(CostPerShipment))   # i
    ShipmentTypes = range(len(Demand))        # type of cargo
    Months = range(1, months + 1)             # t = 1..12

    # Decision Variables
    # ShipmentNum[i, t]: supplier i's number of shipments in month t
    ShipmentNum = model.addVars(
        Suppliers, Months,
        vtype=GRB.INTEGER,
        name="ShipmentNum"
    )

    # Objective: Minimize total cost of shipments over all months and suppliers
    obj = gp.quicksum(
        CostPerShipment[s] * ShipmentNum[s, t]
        for s in Suppliers
        for t in Months
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1 (original per-supplier max shipments) – now commented out
    # ❤ Non-linearity is introduced. ❤
    # for s in Suppliers:
    #     model.addConstr(
    #         ShipmentNum[s] <= MaxShipment[s],
    #         f"MaxShipment_{s+1}"
    #     )

    # New Constraint 1: Seasonal, time-varying capacity with cosine adjustment
    # x_i(t) <= 700 * [1 + 0.1 * cos(pi * t / 6)]
    for s in Suppliers:
        for t in Months:
            seasonal_cap = MaxShipmentNominal * (
                1.0 + 0.1 * math.cos(math.pi * t / 6.0)
            )
            # Because cos term and all coefficients are constants for a given t,
            # this remains a linear constraint with a time-varying RHS.
            model.addConstr(
                ShipmentNum[s, t] <= seasonal_cap,
                name=f"SeasonalCapacity_s{s+1}_t{t}"
            )

    # Constraint 2: Monthly demand satisfaction constraint for each cargo type
    # For each month t and cargo type k, sum_i (percent_ik * x_i(t)) >= Demand_k
    for t in Months:
        for k in ShipmentTypes:
            model.addConstr(
                gp.quicksum(
                    (Percent[s][k] / 100.0) * ShipmentNum[s, t]
                    for s in Suppliers
                ) >= Demand[k],
                name=f"DemandSatisfaction_type{k+1}_month{t}"
            )

    # Optimize the model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        # collect solution details
        solution = {
            "status": "optimal",
            "obj": model.objVal,
            "shipments": {
                f"s{s+1}_m{t}": int(ShipmentNum[s, t].X)
                for s in Suppliers
                for t in Months
            }
        }
        return solution
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_supplier_shipment_optimization()
    print(result)
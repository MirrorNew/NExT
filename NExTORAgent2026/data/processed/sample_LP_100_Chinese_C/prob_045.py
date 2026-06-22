import gurobipy as gp
from gurobipy import GRB


def solve_supplier_shipment_optimization(
    CostPerShipment=[5.2, 4.7, 3.5],
    Percent=[
        [45, 35, 20],
        [30, 45, 25],
        [15, 20, 65]
    ],
    Demand=[500, 300, 300],
    MaxShipment=[700, 700, 700],
    ExtraFixedCost=2000,
    Threshold=400
):
    # Create a new model
    model = gp.Model("Supplier Shipment Optimization with Nonlinear-like Fixed Cost")

    # Sets
    Suppliers = range(len(CostPerShipment))
    ShipmentTypes = range(len(Demand))

    # Decision Variables
    ShipmentNum = model.addVars(Suppliers, vtype=GRB.INTEGER, name="ShipmentNum")

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # obj = gp.quicksum(CostPerShipment[s] * ShipmentNum[s] for s in Suppliers)
    # model.setObjective(obj, GRB.MINIMIZE)

    # Binary variables to capture whether a supplier's shipments exceed the threshold
    OverThreshold = model.addVars(Suppliers, vtype=GRB.BINARY, name="OverThreshold")

    # Big-M value: cannot exceed MaxShipment[s] anyway, so use it safely
    # This will linearize the "if ShipmentNum[s] > Threshold then pay ExtraFixedCost" logic
    M = max(MaxShipment)

    # New objective: variable shipment cost + extra fixed cost when exceeding threshold
    obj = gp.quicksum(CostPerShipment[s] * ShipmentNum[s] for s in Suppliers) + \
          gp.quicksum(ExtraFixedCost * OverThreshold[s] for s in Suppliers)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: Shipment number constraint (cannot exceed maximum)
    for s in Suppliers:
        model.addConstr(
            ShipmentNum[s] <= MaxShipment[s],
            f"MaxShipment_{s+1}"
        )

    # Constraint 2: Demand satisfaction constraint
    for t in ShipmentTypes:
        # Convert percentage to decimal
        model.addConstr(
            gp.quicksum((Percent[s][t] / 100) * ShipmentNum[s] for s in Suppliers) >= Demand[t],
            f"DemandSatisfaction_{t+1}"
        )

    # ❤ Non-linearity is introduced. ❤
    # Linearization of the step fixed cost:
    # OverThreshold[s] = 1  ⇒  ShipmentNum[s] can be > Threshold (up to MaxShipment)
    # OverThreshold[s] = 0  ⇒  ShipmentNum[s] ≤ Threshold
    for s in Suppliers:
        # If OverThreshold[s] = 0, this gives ShipmentNum[s] <= Threshold
        # If OverThreshold[s] = 1, this gives ShipmentNum[s] <= Threshold + M,
        # which is non-binding due to MaxShipment constraint
        model.addConstr(
            ShipmentNum[s] <= Threshold + M * OverThreshold[s],
            f"ThresholdImplication_{s+1}"
        )

        # To prevent paying the fixed cost when ShipmentNum[s] <= Threshold,
        # we also ensure that if ShipmentNum[s] > Threshold, OverThreshold[s] must be 1:
        # ShipmentNum[s] >= Threshold + 1 - M * (1 - OverThreshold[s])
        model.addConstr(
            ShipmentNum[s] >= Threshold + 1 - M * (1 - OverThreshold[s]),
            f"ThresholdActivation_{s+1}"
        )

    # Optimize the model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        # Extract shipment numbers and whether fixed cost is triggered
        shipment_solution = {f"Supplier_{s+1}": int(ShipmentNum[s].X) for s in Suppliers}
        fixed_cost_flags = {f"Supplier_{s+1}": int(OverThreshold[s].X) for s in Suppliers}
        return {
            "status": "optimal",
            "obj": model.objVal,
            "ShipmentNum": shipment_solution,
            "OverThreshold": fixed_cost_flags
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_supplier_shipment_optimization()
    print(result)
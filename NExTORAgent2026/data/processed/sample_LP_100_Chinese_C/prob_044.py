import gurobipy as gp
from gurobipy import GRB


def solve_mining_optimization(
    mines=['M1', 'M2', 'M3', 'M4'],
    plants=['P1', 'P2', 'P3'],
    mine_supply={'M1': 6000, 'M2': 9000, 'M3': 7000, 'M4': 5000},
    plant_capacity={'P1': 8000, 'P2': 10000, 'P3': 12000},
    recovery_rate={'M1': 0.85, 'M2': 0.92, 'M3': 0.78, 'M4': 0.88},
    processing_cost={'M1': 12, 'M2': 18, 'M3': 10, 'M4': 15},
    sulfur_content={'M1': 1.2, 'M2': 0.8, 'M3': 2.1, 'M4': 1.5},
    max_sulfur=1.5,
    budget=500000,
    transport_cost={
        'M1': {'P1': 4, 'P2': 6, 'P3': 5},
        'M2': {'P1': 7, 'P2': 5, 'P3': 8},
        'M3': {'P1': 3, 'P2': 4, 'P3': 6},
        'M4': {'P1': 8, 'P2': 6, 'P3': 7}
    },
    # parameters for the non-linear (now linearized) operating rule
    total_tonnage_threshold=24000,
    extra_fixed_cost=50000
):
    """Solve the mining company's ore allocation optimization problem with
    an additional fixed cost when total throughput exceeds a threshold.
    """
    # Create a new model
    model = gp.Model("MiningAllocation")

    # Decision variables: x_mp = tons shipped from mine m to plant p
    x = model.addVars(mines, plants, name="Allocation", lb=0)

    # Total processed tonnage across all plants
    total_tonnage = gp.quicksum(x[m, p] for m in mines for p in plants)

    # ❤ Non-linearity is introduced. ❤
    # Binary variable y indicates whether the extra environmental & safety
    # measures are activated (1 if total_tonnage > threshold, else can be 0)
    y = model.addVar(vtype=GRB.BINARY, name="ExtraMeasures")

    # Objective: Maximize total metal recovery
    model.setObjective(
        gp.quicksum(recovery_rate[m] * x[m, p] for m in mines for p in plants),
        GRB.MAXIMIZE
    )

    # Constraints
    # 1. Mine supply limit
    model.addConstrs(
        (gp.quicksum(x[m, p] for p in plants) <= mine_supply[m] for m in mines),
        name="MineSupply"
    )

    # 2. Plant capacity limit
    model.addConstrs(
        (gp.quicksum(x[m, p] for m in mines) <= plant_capacity[p] for p in plants),
        name="PlantCapacity"
    )

    # 3. Total cost limit (processing + transportation + possible extra fixed cost)
    # ❤ Non-linearity is introduced. ❤
    # Original purely variable cost constraint (commented out):
    # model.addConstr(
    #     gp.quicksum((processing_cost[m] + transport_cost[m][p]) * x[m, p]
    #                for m in mines for p in plants) <= budget,
    #     name="TotalCost"
    # )
    #
    # New cost constraint including the fixed extra cost when y = 1:
    variable_cost = gp.quicksum(
        (processing_cost[m] + transport_cost[m][p]) * x[m, p]
        for m in mines for p in plants
    )
    model.addConstr(
        variable_cost + extra_fixed_cost * y <= budget,
        name="TotalCostWithFixed"
    )

    # 4. Average sulfur content restriction
    total_sulfur = gp.quicksum(sulfur_content[m] * x[m, p] for m in mines for p in plants)
    model.addConstr(total_sulfur <= max_sulfur * total_tonnage, name="SulfurLimit")

    # 5. Logic linking total_tonnage and activation of extra measures (y)
    # We want: if total_tonnage > threshold then y = 1, else y can be 0.
    # This is modeled with a "big-M" style linkage:
    #   total_tonnage <= threshold + M * y
    # When y = 0  ->  total_tonnage <= threshold
    # When y = 1  ->  total_tonnage can exceed threshold (up to M + threshold)
    #
    # Choose a safe M as the maximum physically possible throughput:
    max_possible_tonnage = sum(mine_supply[m] for m in mines)
    M = max_possible_tonnage  # big-M
    model.addConstr(
        total_tonnage <= total_tonnage_threshold + M * y,
        name="ThroughputFixedCostLogic"
    )

    # Optimize the model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "total_tonnage": total_tonnage.getValue(),
            "extra_measures_activated": int(y.getAttr("X")),
        }
    else:
        return {"status": f"{model.status}"}


# Execute the function
if __name__ == "__main__":
    result = solve_mining_optimization()
    print(result)
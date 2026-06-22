import gurobipy as gp
from gurobipy import GRB


def solve_energy_drink_allocation(
    stores=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    customer_traffic={
        'S1': 1200,
        'S2': 800,
        'S3': 1500,
        'S4': 900,
        'S5': 1100,
        'S6': 700
    },
    shelf_space={
        'S1': 50,
        'S2': 30,
        'S3': 60,
        'S4': 35,
        'S5': 45,
        'S6': 25
    },
    similarity_score={
        'S1': 0.8,
        'S2': 0.6,
        'S3': 0.9,
        'S4': 0.7,
        'S5': 0.75,
        'S6': 0.5
    },
    total_inventory=300,
    budget=120,
    standard_cost=5,
    promotional_cost=15,
    space_multiplier=1.2,
    extra_replenish_cost=40
):
    """
    Solves the energy drink allocation problem using Gurobi optimizer.
    Maximizes expected product exposure under inventory, budget, and shelf space constraints,
    with an additional fixed replenishment organization cost if allocated units exceed
    80% of the original shelf capacity at any store.
    """
    # Create model
    model = gp.Model("EnergyDrinkAllocation")

    # Decision variables
    allocate_units = model.addVars(stores, vtype=GRB.INTEGER, name="units")
    display_type = model.addVars(stores, vtype=GRB.BINARY, name="display")

    # ❤ Non-linearity is introduced. ❤
    # Binary variable indicating whether allocated units exceed 80% of original shelf space
    exceed_threshold = model.addVars(stores, vtype=GRB.BINARY, name="exceed80")

    # Objective: Maximize expected exposure
    exposure = gp.quicksum(customer_traffic[s] * allocate_units[s] *
                           similarity_score[s] for s in stores)
    model.setObjective(exposure, GRB.MAXIMIZE)

    # Constraints
    # Inventory limit
    model.addConstr(
        gp.quicksum(allocate_units[s] for s in stores) <= total_inventory,
        "inventory_limit")

    # Budget constraint
    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(
    #     gp.quicksum(standard_cost * (1 - display_type[s]) +
    #                 promotional_cost * display_type[s]
    #                 for s in stores) <= budget, "budget_limit")
    # New budget constraint including extra fixed replenishment cost when threshold is exceeded
    model.addConstr(
        gp.quicksum(
            standard_cost * (1 - display_type[s]) +
            promotional_cost * display_type[s] +
            extra_replenish_cost * exceed_threshold[s]
            for s in stores
        ) <= budget,
        "budget_limit_with_replenish"
    )

    # Shelf space constraints (including promotional expansion)
    for s in stores:
        model.addConstr(
            allocate_units[s] <= shelf_space[s] *
            (1 + (space_multiplier - 1) * display_type[s]),
            f"shelf_space_{s}")

    # ❤ Non-linearity is introduced. ❤
    # Linking constraints to trigger the exceed_threshold binary variable when
    # allocated units go beyond 80% of original shelf space.
    # We linearize the condition:
    # exceed_threshold[s] = 1 if allocate_units[s] > 0.8 * shelf_space[s], else 0
    # using a big-M formulation.
    big_M = max(shelf_space.values())  # sufficient upper bound on units per store
    for s in stores:
        threshold = 0.8 * shelf_space[s]

        # If exceed_threshold[s] == 0  =>  allocate_units[s] <= threshold
        model.addConstr(
            allocate_units[s] <= threshold + big_M * exceed_threshold[s],
            f"exceed_upper_{s}"
        )

        # If allocate_units[s] > threshold, exceed_threshold[s] must be 1.
        # allocate_units[s] >= threshold + 1 - M * (1 - exceed_threshold[s])
        # When exceed_threshold[s] = 0: allocate_units[s] >= threshold + 1 - M (non-binding)
        # When exceed_threshold[s] = 1: allocate_units[s] >= threshold + 1 (forces exceed)
        model.addConstr(
            allocate_units[s] >= threshold + 1 - big_M * (1 - exceed_threshold[s]),
            f"exceed_lower_{s}"
        )

    # Optimize model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        # Optional: you could also return allocation and which stores trigger extra cost
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "allocation": {s: allocate_units[s].X for s in stores},
            "display_type": {s: int(display_type[s].X) for s in stores},
            "exceed_80pct": {s: int(exceed_threshold[s].X) for s in stores}
        }
    else:
        return {"status": f"{model.status}"}


# Execute the function
if __name__ == "__main__":
    result = solve_energy_drink_allocation()
    print(result)
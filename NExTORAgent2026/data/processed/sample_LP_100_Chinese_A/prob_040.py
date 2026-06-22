import gurobipy as gp
from gurobipy import GRB


def solve_energy_drink_allocation():
    stores = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    customer_traffic = {
        'S1': 1200,
        'S2': 800,
        'S3': 1500,
        'S4': 900,
        'S5': 1100,
        'S6': 700
    }
    shelf_space = {
        'S1': 50,
        'S2': 30,
        'S3': 60,
        'S4': 35,
        'S5': 45,
        'S6': 25
    }
    similarity_score = {
        'S1': 0.8,
        'S2': 0.6,
        'S3': 0.9,
        'S4': 0.7,
        'S5': 0.75,
        'S6': 0.5
    }
    total_inventory = 300
    budget = 120
    standard_cost = 5
    promotional_cost = 15
    space_multiplier = 1.2
    k = 0.5
    """
    Solves the energy drink allocation problem using Gurobi optimizer.
    Maximizes expected product exposure under inventory, budget, and shelf space constraints,
    with a non-linear coupled exposure function.
    """
    # Create model
    model = gp.Model("EnergyDrinkAllocation")

    # Decision variables
    allocate_units = model.addVars(stores, vtype=GRB.INTEGER, name="units")
    display_type = model.addVars(stores, vtype=GRB.BINARY, name="display")

    # New binary decision variables: whether to enable small in-store advertising per store
    ad_promo = model.addVars(stores, vtype=GRB.BINARY, name="adPromo")

    # Objective: Maximize expected exposure (non-linear, coupled relationship)
    # ❤ Non-linearity is introduced. ❤
    # exposure = gp.quicksum(customer_traffic[s] * allocate_units[s] *
    #                        similarity_score[s] for s in stores)
    Y = model.addVars(stores, vtype=GRB.CONTINUOUS, name="Y")
    for s in stores:
        model.addConstr(Y[s] == display_type[s] * allocate_units[s])
    exposure = gp.quicksum(
        k
        * customer_traffic[s]
        * Y[s]
        * similarity_score[s]
        * ad_promo[s]
        for s in stores
    )
    model.setObjective(exposure, GRB.MAXIMIZE)

    # Constraints
    # Inventory limit
    model.addConstr(
        gp.quicksum(allocate_units[s] for s in stores) <= total_inventory,
        "inventory_limit")

    # Budget constraint (kept structurally the same; ad_promo assumed to have no extra cost here)
    model.addConstr(
        gp.quicksum(standard_cost * (1 - display_type[s]) +
                    promotional_cost * display_type[s]
                    for s in stores) <= budget, "budget_limit")

    # Shelf space constraints (unchanged)
    for s in stores:
        model.addConstr(
            allocate_units[s] <= shelf_space[s] *
            (1 + (space_multiplier - 1) * display_type[s]),
            f"shelf_space_{s}")

    # Optimize model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        solution = {
            "status": "optimal",
            "objective_value": model.ObjVal,
            "allocation": {s: allocate_units[s].X for s in stores},
            "display_type": {s: display_type[s].X for s in stores},
            "ad_promo": {s: ad_promo[s].X for s in stores},
        }
        return solution
    else:
        return {"status": f"{model.status}"}


# Execute the function
if __name__ == "__main__":
    result = solve_energy_drink_allocation()
    print(result)
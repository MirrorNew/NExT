def optimize_bottle_production(
    water_available=250000,
    min_glass_bottles=20,
    glass_bottle_volume=500,
    plastic_bottle_volume=750,
    plastic_ratio=3,
    fixed_cost=2000,
    threshold_bottles=300
):
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("WaterBottleOptimization_WithFixedCost")
    m.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Decision variables
    x = m.addVar(name="glass_bottles", vtype=GRB.INTEGER, lb=min_glass_bottles)
    y = m.addVar(name="plastic_bottles", vtype=GRB.INTEGER, lb=plastic_ratio * min_glass_bottles)

    # Binary variable to indicate whether total bottles exceed the threshold
    z = m.addVar(name="fixed_cost_trigger", vtype=GRB.BINARY)

    # Total bottles variable (integer, but could also be continuous since it's x + y)
    total_bottles = m.addVar(name="total_bottles", vtype=GRB.INTEGER)

    # Link total_bottles with x and y
    m.addConstr(total_bottles == x + y, "TotalBottlesDef")

    # ------------------------------------------------------------
    # Objective: originally maximize total bottles
    # m.setObjective(x + y, GRB.MAXIMIZE)
    # ❤ Non-linearity is introduced. ❤
    # New objective: maximize (total bottles − penalty if total bottles > threshold).
    # Here we treat it as a profit-like metric: each bottle has unit "benefit",
    # and we subtract the fixed operating cost when z = 1.
    m.setObjective(total_bottles - fixed_cost * z, GRB.MAXIMIZE)
    # ------------------------------------------------------------

    # Add water constraint
    m.addConstr(glass_bottle_volume * x + plastic_bottle_volume * y <= water_available, "WaterLimit")

    # Add ratio constraint: y >= 3x
    m.addConstr(y >= plastic_ratio * x, "PlasticRatio")

    # -------------------------------------------------------------------
    # Logic to model: if total_bottles > threshold_bottles then z = 1,
    # else z = 0. We linearize this using big-M constraints.
    # Choose a sufficiently large M: maximum possible bottles with the water.
    max_possible_bottles = water_available // min(glass_bottle_volume, plastic_bottle_volume)

    # total_bottles <= threshold + M * z
    m.addConstr(
        total_bottles <= threshold_bottles + max_possible_bottles * z,
        "UpperLink_z"
    )

    # total_bottles >= threshold + 1 - M * (1 - z)
    # Ensures that when z = 1, total_bottles can exceed threshold;
    # when z = 0, total_bottles <= threshold.
    m.addConstr(
        total_bottles >= threshold_bottles + 1 - max_possible_bottles * (1 - z),
        "LowerLink_z"
    )
    # -------------------------------------------------------------------

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        glass_solution = int(x.X)
        plastic_solution = int(y.X)
        total_bottles_solution = int(total_bottles.X)
        fixed_cost_paid = int(z.X) * fixed_cost
        objective_value = m.objVal

        return {
            "glass_bottles": glass_solution,
            "plastic_bottles": plastic_solution,
            "total_bottles": total_bottles_solution,
            "fixed_cost_paid": fixed_cost_paid,
            "objective_value": objective_value,
        }
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_bottle_production()
    if result is not None:
        print(f"Glass Bottles: {result['glass_bottles']}")
        print(f"Plastic Bottles: {result['plastic_bottles']}")
        print(f"Total Bottles: {result['total_bottles']}")
        print(f"Fixed Cost Paid: {result['fixed_cost_paid']}")
        print(f"Objective Value (Total Bottles - Fixed Cost if any): {result['objective_value']}")
    else:
        print("No feasible solution found.")
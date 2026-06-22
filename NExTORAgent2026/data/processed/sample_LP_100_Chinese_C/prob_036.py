import gurobipy as gp
from gurobipy import GRB
import math


def solve_bakery_distribution(
    BakerySupply=[200, 150, 250, 180],
    StoreDemand=[120, 100, 130, 90, 80],
    ShippingDistance=[[10, 15, 20, 25, 30],
                      [12, 8, 18, 22, 28],
                      [14, 10, 16, 20, 26],
                      [16, 12, 14, 18, 24]]):
    """
    Solves the bakery distribution (transportation) problem with
    a non-linear (fixed charge) cost component:

    - Base cost: sqrt(distance) dollars per loaf.
    - Additional rule: If more than 150 loaves are shipped on a specific
      bakery-to-store route in a day, a fixed cost of 200 dollars is added
      to the total cost of that route.
    """
    # --- 1. Model Creation ---
    model = gp.Model("Bakery Distribution Optimization")

    # --- 2. Parameters & Sets ---
    BakeryNum = len(BakerySupply)
    StoreNum = len(StoreDemand)
    Bakeries = range(BakeryNum)
    Stores = range(StoreNum)

    # Calculate shipping costs with square roots (per loaf cost)
    ShippingCost = [[math.sqrt(dist) for dist in row]
                    for row in ShippingDistance]

    # --- 3. Decision Variables ---

    # Amount shipped (integer loaves) from bakery b to store s
    ShipAmount = model.addVars(Bakeries,
                               Stores,
                               vtype=GRB.INTEGER,
                               name="ShipAmount")

    # ❤ Non-linearity is introduced. ❤
    # Binary variable indicating whether the route (b, s) pays the extra fixed cost
    FixedCostUsed = model.addVars(Bakeries,
                                  Stores,
                                  vtype=GRB.BINARY,
                                  name="FixedCostUsed")

    # --- 4. Objective Function ---
    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # model.setObjective(
    #     gp.quicksum(ShippingCost[b][s] * ShipAmount[b, s]
    #                 for b in Bakeries for s in Stores),
    #     GRB.MINIMIZE)

    # New objective:
    #   Variable cost: sqrt(distance) * ShipAmount[b,s]
    #   Fixed cost:   200 * FixedCostUsed[b,s]
    model.setObjective(
        gp.quicksum(
            ShippingCost[b][s] * ShipAmount[b, s] +
            200 * FixedCostUsed[b, s]
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

    # --- 5b. Non-linear fixed-charge logic (modeled linearly) ---
    # If more than 150 loaves are shipped on route (b,s),
    # then FixedCostUsed[b,s] must be 1, and 200 dollars are added in the objective.
    #
    # We implement:
    #   ShipAmount[b,s] - 150 <= M * FixedCostUsed[b,s]
    #
    # - If ShipAmount[b,s] > 150, the LHS > 0, so FixedCostUsed[b,s] must be 1.
    # - If ShipAmount[b,s] <= 150, the constraint is non-binding, so the model
    #   can freely choose FixedCostUsed[b,s] = 0 to avoid the extra 200.
    #
    # Choose a sufficiently large M (upper bound on ShipAmount[b,s]).
    # A safe bound is total demand.
    total_demand = sum(StoreDemand)
    M = total_demand

    # ❤ Non-linearity is introduced. ❤
    model.addConstrs(
        (ShipAmount[b, s] - 150 <= M * FixedCostUsed[b, s]
         for b in Bakeries for s in Stores),
        name="FixedCostTrigger"
    )

    # --- 6. Solve the Model ---
    model.optimize()

    # --- 7. Return Results ---
    if model.status == GRB.OPTIMAL:
        solution = {
            "status": "optimal",
            "obj": model.ObjVal,
            "shipments": {},
            "fixed_cost_used": {}
        }
        for b in Bakeries:
            for s in Stores:
                qty = ShipAmount[b, s].X
                if qty > 0:
                    solution["shipments"][(b, s)] = qty
                    solution["fixed_cost_used"][(b, s)] = FixedCostUsed[b, s].X
        return solution
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_bakery_distribution()
    print(result)
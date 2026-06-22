import gurobipy as gp
from gurobipy import GRB


def solve_greentech_manufacturing(
):
    cost_per_ton_method={"Method1": 4.20, "Method2": 3.80, "Method3": 4.50}
    yield_per_method={
        "Method1": {"Q6": 0.3, "Q8": 0.4, "Q10": 0.3},
        "Method2": {"Q6": 0.5, "Q8": 0.3, "Q10": 0.2},
        "Method3": {"Q6": 0.2, "Q8": 0.5, "Q10": 0.3}
    }
    actual_quality_values={"Q6": 6, "Q8": 8, "Q10": 10}
    upgrade_cost={"Q6_to_Q8": 1.20, "Q8_to_Q10": 1.80}
    price_per_ton={"Premium": 15, "Standard": 8}
    min_quality_score_req={"Premium": 9, "Standard": 7}
    max_demand={"Premium": 2500, "Standard": 800}
    """
    Models and solves the GreenTech Manufacturing problem
    with a non-linear profit term for premium packaging.
    """
    # Create a new model
    model = gp.Model("GreenTechManufacturing_Nonlinear")

    # --- Sets ---
    packaging_types = ["Premium", "Standard"]
    methods = ["Method1", "Method2", "Method3"]
    quality_labels = ["Q6", "Q8", "Q10"]

    # --- Original Decision Variables (kept for linear structure) ---
    # Processing amount per method
    processing_amount = model.addVars(methods, name="ProcessingAmount", lb=0)
    # Upgrade amounts
    quality6_upgrade_amount = model.addVar(name="Quality6UpgradeAmount", lb=0)
    quality8_upgrade_amount = model.addVar(name="Quality8UpgradeAmount", lb=0)
    # Packaging produced by type and quality level
    packaging_produced = model.addVars(packaging_types, quality_labels,
                                       name="PackagingProduced", lb=0)

    # --- New Decision Variables corresponding to the modified description ---
    # x1, x2, x3: tons processed by methods 1, 2, 3
    x1 = model.addVar(name="x1", lb=0)
    x2 = model.addVar(name="x2", lb=0)
    x3 = model.addVar(name="x3", lb=0)

    # yH, yS: total premium and standard packaging tons
    yH = model.addVar(name="yH", lb=0)
    yS = model.addVar(name="yS", lb=0)


    # t: flexibility coefficient in [0, 2]
    t = model.addVar(name="t", lb=0, ub=2)

    # --- Link new variables to original structure (linear consistency) ---
    # Link x1, x2, x3 to processing_amount
    model.addConstr(x1 == processing_amount["Method1"], name="Link_x1_Method1")
    model.addConstr(x2 == processing_amount["Method2"], name="Link_x2_Method2")
    model.addConstr(x3 == processing_amount["Method3"], name="Link_x3_Method3")

    # Link yH, yS to packaging_produced sums
    model.addConstr(
        yH == gp.quicksum(packaging_produced["Premium", q] for q in quality_labels),
        name="Link_yH_PremiumTotal"
    )
    model.addConstr(
        yS == gp.quicksum(packaging_produced["Standard", q] for q in quality_labels),
        name="Link_yS_StandardTotal"
    )

    # --- Linear part of Objective Function: Total Revenue - Costs ---

    # ❤ Non-linearity is introduced. ❤
    # total_revenue = gp.quicksum(price_per_ton[p] * packaging_produced.sum(p, '*') for p in packaging_types)
    total_revenue = gp.quicksum(
        price_per_ton[p] * gp.quicksum(packaging_produced[p, q] for q in quality_labels)
        for p in packaging_types
    )

    total_processing_cost = gp.quicksum(cost_per_ton_method[m] * processing_amount[m] for m in methods)
    total_upgrade_cost = (
        upgrade_cost["Q6_to_Q8"] * quality6_upgrade_amount +
        upgrade_cost["Q8_to_Q10"] * quality8_upgrade_amount
    )

    linear_profit_part = total_revenue - total_processing_cost - total_upgrade_cost

    # --- Non-linear core profit term for premium packaging ---
    # High-end (premium) core profit = 0.002 * yH * x1 * z * t
    Y = model.addVar()
    model.addConstr(Y == yH * x1 )
    core_premium_profit = 0.00001 * Y * t

    # --- Combined (non-linear) Objective ---
    model.setObjective(linear_profit_part + core_premium_profit, GRB.MAXIMIZE)

    # --- Constraints (kept linear, same structure as original) ---

    # Quality constraints for each packaging type
    for p in packaging_types:
        weighted_quality_sum = gp.quicksum(
            actual_quality_values[q_label] * packaging_produced[p, q_label]
            for q_label in quality_labels
        )
        total_produced_p = gp.quicksum(packaging_produced[p, q] for q in quality_labels)
        model.addConstr(
            weighted_quality_sum >= min_quality_score_req[p] * total_produced_p,
            name=f"QualityScore_{p}"
        )

    # Demand constraints for each packaging type
    for p in packaging_types:
        model.addConstr(
            gp.quicksum(packaging_produced[p, q] for q in quality_labels) <= max_demand[p],
            name=f"Demand_{p}"
        )

    # Material balance for Q6
    produced_q6_material = gp.quicksum(
        yield_per_method[m]["Q6"] * processing_amount[m] for m in methods
    )
    used_as_q6_in_packaging = gp.quicksum(packaging_produced[p, "Q6"] for p in packaging_types)
    model.addConstr(
        produced_q6_material - quality6_upgrade_amount >= used_as_q6_in_packaging,
        name="Balance_Q6"
    )

    # Material balance for Q8
    produced_q8_material_native = gp.quicksum(
        yield_per_method[m]["Q8"] * processing_amount[m] for m in methods
    )
    total_available_q8_material = produced_q8_material_native + quality6_upgrade_amount
    used_as_q8_in_packaging = gp.quicksum(packaging_produced[p, "Q8"] for p in packaging_types)
    model.addConstr(
        total_available_q8_material - quality8_upgrade_amount >= used_as_q8_in_packaging,
        name="Balance_Q8"
    )

    # Material balance for Q10
    produced_q10_material_native = gp.quicksum(
        yield_per_method[m]["Q10"] * processing_amount[m] for m in methods
    )
    total_available_q10_material = produced_q10_material_native + quality8_upgrade_amount
    used_as_q10_in_packaging = gp.quicksum(packaging_produced[p, "Q10"] for p in packaging_types)
    model.addConstr(
        total_available_q10_material >= used_as_q10_in_packaging,
        name="Balance_Q10"
    )

    # Optimize the model (this is now a non-linear model)
    model.Params.NonConvex = 2  # allow non-convex quadratic / polynomial terms
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "x1": x1.X,
            "x2": x2.X,
            "x3": x3.X,
            "yH": yH.X,
            "yS": yS.X,
            "t": t.X
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == '__main__':
    result = solve_greentech_manufacturing()
    print(result)
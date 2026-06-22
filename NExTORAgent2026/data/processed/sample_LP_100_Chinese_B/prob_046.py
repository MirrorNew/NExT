import gurobipy as gp
from gurobipy import GRB


def solve_fleet_optimization(
    avg_carbon_emission_limit=40,
    revenue_per_vehicle=[25000, 20000],
    operating_cost_per_vehicle=[18000, 16000],
    carbon_emission_per_vehicle=[70, 30],
    max_vehicles_per_year=[[300, 320, 350], [250, 280, 300]],
    max_total_vehicles=400
):
    """
    Models and solves the fleet optimization problem.
    """
    # Create a new model
    model = gp.Model("Fleet_Optimization")

    # Define sets
    V = range(2)  # Vehicle types: 0=Type X, 1=Type Y
    Y = range(3)  # Years: 0=Year 1, 1=Year 2, 2=Year 3

    # Decision variables
    vehicles_deployed = {}
    for v in V:
        for y in Y:
            vehicles_deployed[v, y] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                name=f"VehiclesDeployed_{v}_{y}"
            )

    # Objective function: Maximize total profit
    profit = gp.quicksum(
        (revenue_per_vehicle[v] - operating_cost_per_vehicle[v]) * vehicles_deployed[v, y]
        for v in V for y in Y
    )
    model.setObjective(profit, GRB.MAXIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Constraint 1: Carbon emissions constraint (replaced by a 3-year average ratio constraint)
    # for y in Y:
    #     total_vehicles = gp.quicksum(vehicles_deployed[v, y] for v in V)
    #     total_emissions = gp.quicksum(carbon_emission_per_vehicle[v] * vehicles_deployed[v, y] for v in V)
    #     model.addConstr(
    #         total_emissions <= avg_carbon_emission_limit * total_vehicles,
    #         name=f"CarbonEmissions_Year_{y}"
    #     )
    total_vehicles_3yr = gp.quicksum(vehicles_deployed[v, y] for v in V for y in Y)
    total_emissions_3yr = gp.quicksum(
        carbon_emission_per_vehicle[v] * vehicles_deployed[v, y] for v in V for y in Y
    )
    # Average emission over all vehicles deployed in 3 years:
    # total_emissions_3yr / total_vehicles_3yr <= avg_carbon_emission_limit
    # This is a nonlinear (fractional) constraint.
    model.addConstr(
        total_emissions_3yr <= avg_carbon_emission_limit * total_vehicles_3yr,
        name="CarbonEmissions_3Year_Average"
    )

    # Constraint 2: Fleet capacity constraint (per year, max 400 vehicles)
    for y in Y:
        model.addConstr(
            gp.quicksum(vehicles_deployed[v, y] for v in V) <= max_total_vehicles,
            name=f"FleetCapacity_Year_{y}"
        )

    # Constraint 3: Yearly vehicle limit constraint per type
    for v in V:
        for y in Y:
            model.addConstr(
                vehicles_deployed[v, y] <= max_vehicles_per_year[v][y],
                name=f"YearlyVehicleLimit_Type_{v}_Year_{y}"
            )

    # Solve the model
    model.optimize()

    # Return Results
    if model.status == GRB.OPTIMAL:
        return {"status": "optimal", "obj": model.ObjVal}
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_fleet_optimization()
    print(result)
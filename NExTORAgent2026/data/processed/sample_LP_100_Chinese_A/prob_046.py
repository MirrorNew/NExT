import gurobipy as gp
from gurobipy import GRB


def solve_fleet_optimization(
):
    avg_carbon_emission_limit=40
    revenue_per_vehicle=[25000, 20000]
    operating_cost_per_vehicle=[18000, 16000]
    carbon_emission_per_vehicle=[70, 30]
    max_vehicles_per_year=[[300, 320, 350], [250, 280, 300]]
    max_total_vehicles=400
    """
    Models and solves the fleet optimization problem with a non-linear
    operating cost model for Type Y vehicles.

    Non-linearity:
        Three-year total operating cost for all Type Y vehicles is:
            16,000 * (Total_Y_vehicles ** 1.2)
    """

    # Create a new model
    model = gp.Model("Fleet_Optimization")

    # Important: allow non-convex quadratic/POW expressions
    model.Params.NonConvex = 2

    # Define sets
    V = range(2)  # Vehicle types: 0=Type X, 1=Type Y
    Y = range(3)  # Years: 0=Year 1, 1=Year 2, 2=Year 3

    # Decision variables: number of vehicles deployed by type and year
    vehicles_deployed = {}
    for v in V:
        for y in Y:
            vehicles_deployed[v, y] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                name=f"VehiclesDeployed_{v}_{y}"
            )

    # Auxiliary variable for total number of Y-type vehicles across 3 years
    total_Y = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Total_Y_Vehicles")

    # Link total_Y with the sum of Y-type vehicles (v=1) across all years
    model.addConstr(
        total_Y == gp.quicksum(vehicles_deployed[1, y] for y in Y),
        name="Total_Y_Definition"
    )

    # ❤ Non-linearity is introduced. ❤
    # Original linear profit expression:
    # profit = gp.quicksum(
    #     (revenue_per_vehicle[v] - operating_cost_per_vehicle[v]) * vehicles_deployed[v, y]
    #     for v in V for y in Y
    # )

    # New objective:
    #   Maximize total revenue from X and Y
    #   minus linear operating cost of X
    #   minus non-linear operating cost of Y:
    #       16,000 * (total_Y ** 1.2)
    total_revenue = gp.quicksum(
        revenue_per_vehicle[v] * vehicles_deployed[v, y] for v in V for y in Y
    )

    # Linear operating cost for X-type vehicles only (v = 0)
    total_operating_cost_X = gp.quicksum(
        operating_cost_per_vehicle[0] * vehicles_deployed[0, y] for y in Y
    )

    # Non-linear operating cost for Y-type vehicles (all 3 years combined)

    YY = model.addVar()
    model.addGenConstrPow(total_Y,YY,1.002)
    nonlinear_operating_cost_Y = 16000 * YY
    profit = total_revenue - total_operating_cost_X - nonlinear_operating_cost_Y
    model.setObjective(profit, GRB.MAXIMIZE)

    # Constraint 1: Carbon emissions constraint (per year)
    for y in Y:
        total_vehicles_year = gp.quicksum(vehicles_deployed[v, y] for v in V)
        total_emissions_year = gp.quicksum(
            carbon_emission_per_vehicle[v] * vehicles_deployed[v, y] for v in V
        )
        model.addConstr(
            total_emissions_year <= avg_carbon_emission_limit * total_vehicles_year,
            name=f"CarbonEmissions_Year_{y}"
        )

    # Constraint 2: Fleet capacity constraint (per year)
    for y in Y:
        model.addConstr(
            gp.quicksum(vehicles_deployed[v, y] for v in V) <= max_total_vehicles,
            name=f"FleetCapacity_Year_{y}"
        )

    # Constraint 3: Yearly vehicle limit constraint
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
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "vehicles_deployed": {
                (int(v), int(y)): int(vehicles_deployed[v, y].X)
                for v in V for y in Y
            },
            "total_Y": total_Y.X
        }
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_fleet_optimization()
    print(result)
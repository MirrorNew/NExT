import gurobipy as gp
from gurobipy import GRB


def solve_fleet_optimization(
    avg_carbon_emission_limit=40,
    revenue_per_vehicle=[25000, 20000],
    operating_cost_per_vehicle=[18000, 16000],
    carbon_emission_per_vehicle=[70, 30],
    max_vehicles_per_year=[[300, 320, 350], [250, 280, 300]],
    max_total_vehicles=400,
    management_threshold=350,
    management_fixed_cost=50000
):
    """
    Models and solves the fleet optimization problem with
    stepwise yearly management cost:
    If total vehicles in a year > management_threshold,
    then a fixed management_fixed_cost is incurred in that year.
    """

    # Create a new model
    model = gp.Model("Fleet_Optimization")

    # Define sets
    V = range(2)  # Vehicle types: 0=Type X, 1=Type Y
    Y = range(3)  # Years: 0=Year 1, 1=Year 2, 2=Year 3

    # Decision variables: number of vehicles of each type in each year
    vehicles_deployed = {}
    for v in V:
        for y in Y:
            vehicles_deployed[v, y] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                name=f"VehiclesDeployed_{v}_{y}"
            )

    # Binary variables for stepwise management cost:
    # 1 if in year y total vehicles > threshold, 0 otherwise
    manage_cost_indicator = {}
    for y in Y:
        manage_cost_indicator[y] = model.addVar(
            vtype=GRB.BINARY,
            name=f"ManageCostIndicator_{y}"
        )

    # Precompute per-vehicle profit
    profit_per_vehicle = [
        revenue_per_vehicle[v] - operating_cost_per_vehicle[v] for v in V
    ]

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (without step cost) is commented out:
    # profit = gp.quicksum(
    #     (revenue_per_vehicle[v] - operating_cost_per_vehicle[v]) * vehicles_deployed[v, y]
    #     for v in V for y in Y
    # )
    # model.setObjective(profit, GRB.MAXIMIZE)

    # New objective: total vehicle profit minus stepwise management costs
    total_profit = gp.quicksum(
        profit_per_vehicle[v] * vehicles_deployed[v, y]
        for v in V for y in Y
    ) - gp.quicksum(
        management_fixed_cost * manage_cost_indicator[y] for y in Y
    )
    model.setObjective(total_profit, GRB.MAXIMIZE)

    # Constraint 1: Carbon emissions constraint (per year average ≤ limit)
    for y in Y:
        total_vehicles_y = gp.quicksum(vehicles_deployed[v, y] for v in V)
        total_emissions_y = gp.quicksum(
            carbon_emission_per_vehicle[v] * vehicles_deployed[v, y] for v in V
        )
        model.addConstr(
            total_emissions_y <= avg_carbon_emission_limit * total_vehicles_y,
            name=f"CarbonEmissions_Year_{y}"
        )

    # Constraint 2: Fleet capacity constraint (max total vehicles per year)
    for y in Y:
        model.addConstr(
            gp.quicksum(vehicles_deployed[v, y] for v in V) <= max_total_vehicles,
            name=f"FleetCapacity_Year_{y}"
        )

    # Constraint 3: Yearly vehicle-type-specific maximums
    for v in V:
        for y in Y:
            model.addConstr(
                vehicles_deployed[v, y] <= max_vehicles_per_year[v][y],
                name=f"YearlyVehicleLimit_Type_{v}_Year_{y}"
            )

    # ❤ Non-linearity is introduced. ❤
    # Add constraints that link the binary step-cost indicator to total vehicles.
    # If total_vehicles_y > threshold, indicator must be 1; if it's 0, total_vehicles_y
    # is forced to be ≤ threshold. This creates the stepwise management cost behavior.
    big_M = max_total_vehicles  # safe upper bound on total vehicles in any year
    for y in Y:
        total_vehicles_y = gp.quicksum(vehicles_deployed[v, y] for v in V)

        # If indicator[y] == 0  ⇒  total_vehicles_y ≤ threshold
        model.addConstr(
            total_vehicles_y <= management_threshold +
            big_M * manage_cost_indicator[y],
            name=f"MgmtCost_UpperLink_Year_{y}"
        )

        # Optional tightening: If total_vehicles_y >= threshold + 1 then indicator[y] must be 1
        # total_vehicles_y ≥ (threshold + 1) * indicator[y]
        model.addConstr(
            total_vehicles_y >= (management_threshold + 1) * manage_cost_indicator[y],
            name=f"MgmtCost_LowerLink_Year_{y}"
        )

    # Solve the model
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        solution = {
            "status": "optimal",
            "obj": model.ObjVal,
            "vehicles_deployed": {
                (v, y): int(vehicles_deployed[v, y].X)
                for v in V for y in Y
            },
            "management_cost_indicator": {
                y: int(manage_cost_indicator[y].X) for y in Y
            }
        }
        return solution
    else:
        return {"status": f"{model.status}"}


if __name__ == "__main__":
    result = solve_fleet_optimization()
    print(result)
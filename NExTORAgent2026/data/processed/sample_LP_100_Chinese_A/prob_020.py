import gurobipy as gp
from gurobipy import GRB


def solve_school_bus_rental():
    """
    Solves the school bus rental problem to minimize total cost,
    subject to student capacity, vehicle availability, and driver constraints.
    Now includes a nonlinear cost term for large buses:
        If x large buses are used, their total cost is (800 x)^1.2.
    """
    try:
        # --- Parameters ---
        num_students = 400

        bus_types = ['LargeBus', 'Minibus']

        # Vehicle details
        capacity = {'LargeBus': 50, 'Minibus': 40}  # seats
        availability = {'LargeBus': 10, 'Minibus': 8}  # number of vehicles
        rental_cost = {'LargeBus': 800, 'Minibus': 600}  # £ per vehicle

        # Driver availability
        available_drivers = 9

        # --- Create Gurobi Model ---
        model = gp.Model("SchoolBusRental_Nonlinear")

        # --- Decision Variables ---
        # N[bt]: Number of buses of type bt to rent
        N = model.addVars(bus_types,
                          name="NumVehicles",
                          vtype=GRB.CONTINUOUS,
                          lb=0)

        # --- Objective Function: Minimize Total Rental Cost ---
        # ❤ Non-linearity is introduced. ❤
        # model.setObjective(
        #     gp.quicksum(rental_cost[bt] * N[bt] for bt in bus_types),
        #     GRB.MINIMIZE)

        # Nonlinear objective:
        #   LargeBus total cost: (800 * N['LargeBus'])^1.2
        #   Minibus total cost: 600 * N['Minibus'] (still linear)

        Y = model.addVar(name="Y", lb=0.0, vtype=GRB.CONTINUOUS)
        eY = model.addVar(name="kY", lb=0.0, vtype=GRB.CONTINUOUS)
        model.addConstr(Y == rental_cost['LargeBus'] * N['LargeBus'])
        model.addGenConstrPow(Y,eY,1.1)

        minibus_cost_expr = rental_cost['Minibus'] * N['Minibus']

        model.setObjective(eY + minibus_cost_expr,
                           GRB.MINIMIZE)

        # --- Constraints ---
        # 1. Student Capacity Constraint: Total seats >= num_students
        model.addConstr(gp.quicksum(capacity[bt] * N[bt] for bt in bus_types)
                        >= num_students,
                        name="StudentCapacity")

        # 2. Driver Availability Constraint: Total vehicles <= available_drivers
        model.addConstr(gp.quicksum(N[bt] for bt in bus_types)
                        <= available_drivers,
                        name="DriverLimit")

        # 3. Vehicle Availability Constraints: N[bt] <= availability[bt]
        for bt in bus_types:
            model.addConstr(N[bt] <= availability[bt],
                            name=f"VehicleLimit_{bt}")

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Since we use a nonlinear objective, we explicitly set the model type
        model.ModelSense = GRB.MINIMIZE

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal bus rental plan found (with nonlinear large-bus cost).")
            print(f"Minimum Total Rental Cost: £{model.ObjVal:.2f}")

            print("\nNumber of Vehicles to Rent:")
            total_vehicles = 0
            total_capacity = 0
            for bt in bus_types:
                print(
                    f"  {bt}: {N[bt].X:.2f} (Max Available: {availability[bt]})"
                )
                total_vehicles += N[bt].X
                total_capacity += capacity[bt] * N[bt].X

            print("\nSummary:")
            print(
                f"  Total Vehicles Rented: {total_vehicles:.2f} (Drivers Available: {available_drivers})"
            )
            print(
                f"  Total Seating Capacity: {total_capacity:.2f} (Students: {num_students})"
            )

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. It's impossible to transport all students with the available vehicles/drivers."
            )
            # Compute and print IIS (Irreducible Inconsistent Subsystem)
            model.computeIIS()
            model.write("bus_rental_iis.ilp")
            print("IIS written to bus_rental_iis.ilp for debugging.")
        else:
            print(f"Optimization stopped with status: {model.status}")
            if model.SolCount == 0:
                print("No feasible solution found.")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_school_bus_rental()
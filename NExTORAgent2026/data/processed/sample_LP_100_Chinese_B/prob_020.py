import gurobipy as gp
from gurobipy import GRB


def solve_school_bus_rental():
    """
    Solves the modified school bus rental problem to minimize total cost,
    subject to student capacity, vehicle availability, driver constraints,
    and an additional nonlinear-type seat ratio condition (implemented in a
    linearized form).
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
        model = gp.Model("SchoolBusRental_WithSeatRatio")

        # --- Decision Variables ---
        # N[bt]: Number of buses of type bt to rent
        N = model.addVars(bus_types,
                          name="NumVehicles",
                          vtype=GRB.INTEGER,
                          lb=0)

        # --- Objective Function: Minimize Total Rental Cost ---
        model.setObjective(
            gp.quicksum(rental_cost[bt] * N[bt] for bt in bus_types),
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

        # 4. Seat ratio constraint:
        #    "使大巴士提供的座位数与中巴士提供的座位数的比值不超过2"
        #    i.e., 50 * N_LargeBus / (40 * N_Minibus) <= 2
        #    which is nonlinear due to the division by N_Minibus.
        #
        #    Algebraically this becomes:
        #       50 * N_LargeBus <= 2 * 40 * N_Minibus
        #       50 * N_LargeBus <= 80 * N_Minibus
        #    which is a linear inequality between integer variables.
        #
        # ❤ Non-linearity is introduced. ❤
        model.addConstr(
            50 * N['LargeBus'] <= 80 * N['Minibus'],
            name="SeatRatio_Large_to_Mini"
        )

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal bus rental plan found.")
            print(f"Minimum Total Rental Cost: £{model.ObjVal:.2f}")

            print("\nNumber of Vehicles to Rent:")
            total_vehicles = 0
            total_capacity = 0
            for bt in bus_types:
                print(
                    f"  {bt}: {N[bt].X:.0f} (Max Available: {availability[bt]})"
                )
                total_vehicles += N[bt].X
                total_capacity += capacity[bt] * N[bt].X

            print("\nSummary:")
            print(
                f"  Total Vehicles Rented: {total_vehicles:.0f} (Drivers Available: {available_drivers})"
            )
            print(
                f"  Total Seating Capacity: {total_capacity:.0f} (Students: {num_students})"
            )

            # Additional summary for the ratio constraint
            large_seats = capacity['LargeBus'] * N['LargeBus'].X
            mini_seats = capacity['Minibus'] * N['Minibus'].X
            if mini_seats > 0:
                ratio = large_seats / mini_seats
                print(
                    f"  Seat Ratio (Large/Minibus): {large_seats:.0f}/{mini_seats:.0f} = {ratio:.3f} (<= 2)"
                )
            else:
                print(
                    "  No Minibuses used; seat ratio Large/Minibus is undefined but constrained by the model."
                )

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. It's impossible to transport all students with the available vehicles/drivers and seat-ratio requirement."
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
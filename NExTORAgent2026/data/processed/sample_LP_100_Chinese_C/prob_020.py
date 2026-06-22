import gurobipy as gp
from gurobipy import GRB


def solve_school_bus_rental():
    """
    Solves the school bus rental problem to minimize total cost,
    subject to student capacity, vehicle availability, and driver constraints.
    Includes non-linear (piecewise / conditional) cost elements modeled via
    binary variables and big-M linearization.
    """
    try:
        # --- Parameters ---
        num_students = 400

        bus_types = ['LargeBus', 'Minibus']

        # Vehicle details
        capacity = {'LargeBus': 50, 'Minibus': 40}  # seats
        availability = {'LargeBus': 10, 'Minibus': 8}  # number of vehicles
        rental_cost = {'LargeBus': 800, 'Minibus': 600}  # base £ per vehicle

        # Driver availability
        available_drivers = 9

        # --- Create Gurobi Model ---
        model = gp.Model("SchoolBusRental")

        # --- Decision Variables ---
        # N[bt]: Number of buses of type bt to rent
        N = model.addVars(bus_types,
                          name="NumVehicles",
                          vtype=GRB.INTEGER,
                          lb=0)

        # Binary variable indicating if we use more than 6 large buses (i.e., at least 7)
        y_large_extra = model.addVar(vtype=GRB.BINARY, name="UseMoreThan6Large")

        # Binary variable indicating if we use at least 5 minibuses
        y_minibus_extra = model.addVar(vtype=GRB.BINARY, name="UseAtLeast5Minibus")

        # --- Objective Function: Minimize Total Rental Cost ---
        # ❤ Non-linearity is introduced. ❤
        # model.setObjective(
        #     gp.quicksum(rental_cost[bt] * N[bt] for bt in bus_types),
        #     GRB.MINIMIZE)

        # Base rental cost for all buses
        base_cost = gp.quicksum(rental_cost[bt] * N[bt] for bt in bus_types)

        # Additional fixed cost: if LargeBus count >= 7, pay extra 2000
        large_bus_extra_fixed_cost = 2000 * y_large_extra

        # Additional per-vehicle cost for minibuses:
        # if Minibus count >= 5, each minibus pays extra 100
        minibus_congestion_cost = 100 * N['Minibus'] * y_minibus_extra

        model.setObjective(base_cost + large_bus_extra_fixed_cost + minibus_congestion_cost,
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

        # --- Non-linear logic encoded via additional linear constraints ---

        # For large buses:
        # y_large_extra = 1  ⇔ N['LargeBus'] >= 7 (within the feasible range 0..10)
        # Implement using big-M bounds consistent with availability.

        # If y_large_extra = 1, force N_LargeBus >= 7
        model.addConstr(N['LargeBus'] >= 7 * y_large_extra,
                        name="LargeBusExtra_Lower")

        # If y_large_extra = 0, force N_LargeBus <= 6
        # N_LargeBus <= 6 + (MaxLarge - 6) * y_large_extra
        # Here MaxLarge = 10 → N_LargeBus <= 6 + 4*y_large_extra
        model.addConstr(N['LargeBus'] <= 6 + (availability['LargeBus'] - 6) * y_large_extra,
                        name="LargeBusExtra_Upper")

        # For minibuses:
        # y_minibus_extra = 1  ⇔ N['Minibus'] >= 5 (within 0..8)

        # If y_minibus_extra = 1, force N_Minibus >= 5
        model.addConstr(N['Minibus'] >= 5 * y_minibus_extra,
                        name="MinibusExtra_Lower")

        # If y_minibus_extra = 0, force N_Minibus <= 4
        # N_Minibus <= 4 + (MaxMini - 4) * y_minibus_extra
        # Here MaxMini = 8 → N_Minibus <= 4 + 4*y_minibus_extra
        model.addConstr(N['Minibus'] <= 4 + (availability['Minibus'] - 4) * y_minibus_extra,
                        name="MinibusExtra_Upper")

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

            # Display non-linear related decisions
            print("\nNon-linear Cost Triggers:")
            print(
                f"  Large bus extra fixed cost applied (>=7 large buses)? {'Yes' if y_large_extra.X > 0.5 else 'No'}"
            )
            print(
                f"  Minibus congestion surcharge applied (>=5 minibuses)? {'Yes' if y_minibus_extra.X > 0.5 else 'No'}"
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
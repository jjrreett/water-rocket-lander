import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from rich import print
from solver import RK4

# units
mm = 0.001
psi = 6890
liter = 0.001  # m3
kg = 1
m = 1
m2 = m * m
m3 = m2 * m
s = 1
s2 = s * s

# constants
g = 9.81 * m / s2  # acceleration due to gravity, m/s^2
gamma = 1.4  # adiabatic constant for air
rho_water = 1000 * kg / m3

# constant params
area_nozzle = 3.14 * (6 * mm) ** 2 / 4

# varying params
pressure_init = 100 * psi
water_fill = 1 * liter
air_fill = 1 * liter
dry_mass = 1 * kg


def water_rocket_simulation(y0, area_nozzle, air_fill, dry_mass):
    initial_water_volume = y0[2] / rho_water

    def odes(t, y):
        height, velocity, mass_water, pressure = y

        if mass_water > 0:
            water_flow = area_nozzle * np.sqrt(2 * pressure / rho_water)
            dm_dt = rho_water * water_flow
            water_volume = mass_water / rho_water
            air_volume = air_fill + initial_water_volume - water_volume
            dpressure = (-gamma * pressure / air_volume) * (water_flow / rho_water)
            thrust = dm_dt * np.sqrt(2 * pressure / rho_water)
        else:
            dm_dt = 0
            dpressure = 0
            thrust = 0

        acceleration = (thrust - (mass_water + dry_mass) * g) / (mass_water + dry_mass)
        return [velocity, acceleration, -dm_dt, dpressure]

    # Initial conditions and time span
    t0 = 0
    t_bound = 5
    h = 0.1

    # Pre-allocate arrays
    num_steps = int((t_bound - t0) / h) + 1  # Calculate number of steps
    results = np.zeros((num_steps + 1, len(y0) + 1))  # Additional column for time

    # Initialize the first row of results
    results[0, 0] = t0  # time
    results[0, 1:] = y0  # initial state

    # Create the RK4 solver instance
    solver = RK4(odes, t0, y0, t_bound, h=0.05)

    i = 1
    while solver.status == "running" and solver.t < t_bound:
        message, t_next, y_next = solver._step_impl()
        results[i, 0] = t_next
        results[i, 1:] = y_next
        if y_next[0] < 0:
            break  # Stop simulation if the rocket crashes
        if y_next[2] < 0:
            break  # Stop simulation when out of fuel
        i += 1

    # Trim results array to actual number of steps computed
    results = results[:i, :]

    score = results[-1, 1] ** 2 + results[-1, 2] ** 2
    return score, results


def run_monte_carlo(num_simulations):
    results = []
    for _ in tqdm(range(num_simulations)):
        # Generate initial conditions
        height = np.random.normal(13, 2) * m  # height in meters
        v_zero_height = 30 * m  # reference height for zero velocity calculation
        velocity = -np.sqrt(max(0, 2 * g * (v_zero_height - height)))
        water_mass = 1 * kg
        pressure = np.random.normal(100, 10) * psi

        y0 = np.array([height, velocity, water_mass, pressure])

        # Simulate using the generated initial conditions
        score, _ = water_rocket_simulation(y0, area_nozzle, air_fill, dry_mass)
        results.append((score, y0))

    # Sort results by score
    results.sort(key=lambda x: x[0])
    return results


def plot_trajectory_generator():
    """Generator for plotting trajectories and velocity profiles of the simulation."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 12))

    ax[0].set_title("Height vs. Time")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Height (m)")

    ax[1].set_title("Velocity vs. Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Velocity (m/s)")

    ax[2].set_title("Pressure vs. Time")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Pressure (psi)")

    while True:
        data = yield
        if data is None:  # Check for termination signal
            break  # Exit loop and show the plot

        label, results = data
        ax[0].plot(results[:, 0], results[:, 1], label=results[-1, 1], alpha=0.3)
        ax[1].plot(results[:, 0], results[:, 2], label=results[-1, 2], alpha=0.3)
        ax[2].plot(results[:, 0], results[:, 4] / psi, label=results[-1, 4], alpha=0.3)

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

    plt.tight_layout()
    plt.show()


import multiprocessing


def worker(num_sims):
    return run_monte_carlo(num_sims)[:10]


def run_all_multi(total_sims, batch_size=10_000):
    num_processes = total_sims // batch_size

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Each process gets batch_size simulations to run
        results = pool.map(worker, [batch_size] * num_processes)

    all_sims = [sim for sublist in results for sim in sublist]

    # all_sims = run_monte_carlo(100_000)
    best_sims = all_sims[:10]
    return best_sims


def run_all_single(total_sims):
    run_monte_carlo(total_sims)[:10]


def main():

    best_sims = run_all_multi(100_000)

    plot_gen = plot_trajectory_generator()
    next(plot_gen)  # Initialize the generator

    for sim in best_sims:
        print(sim)
        score, y0 = sim
        score, results = water_rocket_simulation(y0, area_nozzle, air_fill, dry_mass)
        plot_gen.send((score, results))
    try:
        plot_gen.send(None)
    except StopIteration:
        ...


if __name__ == "__main__":
    main()

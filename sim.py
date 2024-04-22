import matplotlib.pyplot as plt
import numpy as np
from rich import print
from scipy.integrate import solve_ivp
from tqdm import tqdm

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
            acceleration = (thrust / (mass_water + dry_mass)) - g
        else:
            dm_dt = 0
            dpressure = 0
            thrust = 0
            acceleration = -g

        return [velocity, acceleration, -dm_dt, dpressure]

    def hit_ground(t, y):
        return y[0]  # Ground event when height is zero

    hit_ground.terminal = True
    hit_ground.direction = -1

    # Initial conditions and time span
    t0 = 0
    t_bound = 2
    h = 0.05

    sol = solve_ivp(
        odes,
        [t0, t_bound],
        y0,
        method=RK4,
        events=hit_ground,
        # dense_output=True,
        h=h,
    )

    ts = sol.t
    results = sol.y.T

    # Check if the simulation ended because of hitting the ground
    if sol.status == 1:
        # Trim the results at the event
        idx = np.where(ts >= sol.t_events[0][0])[0][0]
        results = results[: idx + 1, :]
        ts = ts[: idx + 1]

    score = abs(results[-1, 2])  # Assuming velocity score or similar
    return score, ts, results


def run_monte_carlo(num_simulations):
    results = []
    for _ in tqdm(range(num_simulations)):
        # Generate initial conditions
        height = np.random.normal(10, 2) * m  # height in meters
        v_zero_height = 25 * m  # reference height for zero velocity calculation
        velocity = -np.sqrt(max(0, 2 * g * (v_zero_height - height)))
        water_mass = 1 * kg
        pressure = np.random.normal(90, 10) * psi

        y0 = np.array([height, velocity, water_mass, pressure])

        # Simulate using the generated initial conditions
        score, _, _ = water_rocket_simulation(y0, area_nozzle, air_fill, dry_mass)
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

        label, ts, results = data
        ax[0].plot(ts, results[:, 0], label=results[-1, 0], alpha=0.3)
        ax[1].plot(ts, results[:, 1], label=results[-1, 1], alpha=0.3)
        ax[2].plot(ts, results[:, 3] / psi, label=results[-1, 3], alpha=0.3)

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

    plt.tight_layout()
    plt.show()


import multiprocessing


def worker(num_sims):
    return run_monte_carlo(num_sims)[:10]


def run_all_multi(total_sims, num_cores=8):
    batch_size = total_sims // num_cores
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
        score, ts, results = water_rocket_simulation(
            y0, area_nozzle, air_fill, dry_mass
        )
        plot_gen.send((score, ts, results))
    try:
        plot_gen.send(None)
    except StopIteration:
        ...


if __name__ == "__main__":
    main()

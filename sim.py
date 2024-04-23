import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from rich import print
from scipy.integrate import solve_ivp, RK45
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

# # constant params
# area_nozzle = 3.14 * (6 * mm) ** 2 / 4
# vol_total = 2 * liter

# varying params
# pressure_init = 100 * psi
# vol_water_0 = 1 * liter
# m_dry = 0.5 * kg
# m_water_0 = vol_water_0 / rho_water

def water_rocket_simulation(y0, area_nozzle, m_dry, vol_total):
    def update_equation(t, y):
        height, velocity, vol_water, pressure = y
        dheight = velocity

        if vol_water <= 0:
            return [dheight, -g, 0, 0]
        
        m_water = vol_water * rho_water
        vol_air = vol_total - vol_water

        v_exit = np.sqrt(2 * pressure / rho_water) 
        dvol_water = -area_nozzle * v_exit
        dm_water = rho_water * dvol_water
        dpressure = gamma * pressure / vol_air * dvol_water 
        thrust = -dm_water * v_exit 
        dvelocity = (thrust / (m_dry + m_water)) - g


        return [dheight, dvelocity, dvol_water, 0]

    def hit_ground(t, y):
        return y[0]  # Ground event when height is zero
    
    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(
        update_equation,
        [0, 15],
        y0,
        method=RK4,
        events=hit_ground,
        h=0.1,
    )

    ts = sol.t
    results = sol.y.T

    # Check if the simulation ended because of hitting the ground
    if sol.status == 1:
        # Trim the results at the event
        idx = np.where(ts >= sol.t_events[0][0])[0][0]
        results = results[: idx + 1, :]
        ts = ts[: idx + 1]

    score = abs(results[-1, 1]) 
    return score, ts, results

def gen_monte_carlo_params(seed=None):
    # constant params
    area_nozzle = 3.14 * (5 * mm) ** 2 / 4
    vol_total = 6 * liter

    # varying params
    vol_water_0 = 2 * liter
    m_dry = 0.5 * kg
    pressure = 120 * psi / 2

    # Generate initial conditions
    # v_zero_height = 25 * m  # reference height for zero velocity calculation
    # velocity = -np.sqrt(max(0, 2 * g * (v_zero_height - height)))

    m_water = rho_water * vol_water_0
    dm_water = rho_water * area_nozzle * np.sqrt(2 * pressure / rho_water)
    thrust = dm_water * np.sqrt(2 * pressure / rho_water)
    delta_v = thrust/dm_water * np.log( (m_dry + m_water) / m_dry ) - g * m_water/dm_water
    m0 = m_water + m_dry 
    t = m_water / dm_water
    h0 = delta_v*t - g/2*t**2 + thrust/dm_water * ((m0 - dm_water*t)/dm_water * np.log((m0 - dm_water*t)/m0) + t)

    rng = np.random.default_rng(seed)
    height = rng.normal(h0/2, 1)  # height in meters
    velocity = rng.normal(-delta_v, delta_v * 0.5)
    y0 = np.array([height, velocity, vol_water_0, pressure])
    params = y0, area_nozzle, m_dry, vol_total
    return params

def run_monte_carlo(num_simulations, seed=None):
    results = []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=2**32, size=num_simulations)
    for seed in tqdm(seeds):
        params = gen_monte_carlo_params(seed)

        # Simulate using the generated initial conditions
        score, _, _ = water_rocket_simulation(*params)
        results.append((score, seed))

    # Sort results by score
    results.sort(key=lambda x: x[0])
    return results

def run_all_single(total_sims, keep_n_best=10, seed=None):
    """Run the monte carlo simulations in single threaded"""
    results = run_monte_carlo(total_sims, seed=seed)
    results.sort(key=lambda x: x[0])
    return results[:keep_n_best]

def run_all_multi(total_sims, num_cores=8, keep_n_best=10, seed=None):
    """Run the monte carlo simulations using multiprocessing pool"""

    batch_size = total_sims // num_cores
    num_processes = total_sims // batch_size

    # Initialize the random number generator with the master seed
    rng = np.random.default_rng(seed)

    # Generate unique seeds for each process from the master RNG
    seeds = rng.integers(low=0, high=2**32, size=num_cores)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Each process gets batch_size simulations to run
        results = pool.starmap(run_all_single, [(batch_size, keep_n_best, seed) for seed in seeds])

    all_sims = [sim for sublist in results for sim in sublist]
    all_sims.sort(key=lambda x: x[0])
    return all_sims[:keep_n_best]

def plot_trajectories(datas):
    """Generator for plotting trajectories and velocity profiles of the simulation."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 12))

    ax[0].set_title("Height vs. Time")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Height (m)")

    ax[1].set_title("Velocity vs. Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Velocity (m/s)")

    ax[2].set_title("Initial Height vs. Initial Velocity")
    ax[2].set_xlabel("Height (m)")
    ax[2].set_ylabel("Velocity (m/s)")
    
    for data in datas:
        label, ts, results = data
        ax[0].plot(ts, results[:, 0], label=results[-1, 0], alpha=0.3)
        ax[1].plot(ts, results[:, 1], label=results[-1, 1], alpha=0.3)

    ax[2].scatter([data[2][0, 0] for data in datas], [data[2][0, 1] for data in datas], alpha=0.3)

    # ax[0].legend()
    ax[1].legend()
    # ax[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    # # constant params
    # area_nozzle = 3.14 * (6 * mm) ** 2 / 4
    # vol_total = 2 * liter

    # # varying params
    # vol_water_0 = 1 * liter
    # m_dry = 0.5 * kg
    # pressure = 120 * psi / 2

    # m_water = rho_water * vol_water_0
    # dm_water = rho_water * area_nozzle * np.sqrt(2 * pressure / rho_water)
    # thrust = dm_water * np.sqrt(2 * pressure / rho_water)
    # delta_v = thrust/dm_water * np.log( (m_dry + m_water) / m_dry ) - g * m_water/dm_water
    # m0 = m_water + m_dry 
    # t = m_water / dm_water

    # h0 = delta_v*t - g/2*t**2 + thrust/dm_water * ((m0 - dm_water*t)/dm_water * np.log((m0 - dm_water*t)/m0) + t)

    # print(h0)
    # print(delta_v)
    # print(t)

    # y0 = np.array([h0/2, -delta_v, vol_water_0, pressure])
    # data = [water_rocket_simulation(y0, area_nozzle, m_dry, vol_total)]

    best_sims = run_all_multi(10_000, keep_n_best=25)
    data = [
        water_rocket_simulation(*gen_monte_carlo_params(sim[1])) for sim in best_sims
    ]
    plot_trajectories(data)


if __name__ == "__main__":
    main()


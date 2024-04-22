import cProfile
import pstats

# Your RocketSimulation code or any other relevant imports here
from sim import run_monte_carlo


# Profile the simulation
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_monte_carlo(1000)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime").print_stats(10)  # Adjust to see more or fewer lines

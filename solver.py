import numpy as np
from scipy.integrate import OdeSolver


class RK4(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h=0.01, vectorized=False, **extraneous):
        super().__init__(fun, t0, y0, t_bound, vectorized, **extraneous)
        self.h = h  # Set a fixed step size

    def _step_impl(self):
        t = self.t
        y = self.y
        h = self.h
        f = self.fun

        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t_new = t + h

        # Update the time and state
        self.t_old = self.t
        self.t = t_new
        self.y_old = self.y
        self.y = y_new

        # Return status, time, and the new state
        return True, t_new, y_new

    def _dense_output_impl(self):
        """Return a continuous solution (not implemented here, just a placeholder)."""
        return None


if __name__ == "__main__":
    # Example of using the RK4 solver
    def fun(t, y):
        return -0.04 * y + 10 * np.exp(-((t - 4) ** 2) / 2)

    # Initial conditions
    t0 = 0
    y0 = np.array([1])
    t_bound = 10

    # Create the RK4 solver instance
    solver = RK4(fun, t0, y0, t_bound)

    # Using the solver to step through the solution
    while solver.status == "running":
        message, t_next, y_next = solver._step_impl()
        print(f"Time: {t_next}, State: {y_next}")

        if t_next >= t_bound:
            break

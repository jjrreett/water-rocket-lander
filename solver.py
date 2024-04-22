import numpy as np
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.common import warn_extraneous
from scipy.integrate import DenseOutput


class SimpleDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, y):
        self.t_old = t_old
        self.t = t
        self.y_old = y_old
        self.y = y

    def _call_impl(self, t):
        """Linear interpolation for demonstration purposes."""
        theta = (t - self.t_old) / (self.t - self.t_old)
        return (1 - theta) * self.y_old + theta * self.y


class RK4(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h=0.01, vectorized=False, **extraneous):
        if extraneous:
            warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)
        self.h = h  # Set a fixed step size

    def _step_impl(self):
        try:
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

            if np.isnan(y_new).any() or np.isinf(y_new).any():
                return False, "Numerical instability detected."

            self.t_old = self.t
            self.t = t_new
            self.y_old = self.y
            self.y = y_new

            return True, None
        except Exception as e:
            return False, str(e)

    def _dense_output_impl(self):
        return SimpleDenseOutput(self.t_old, self.t, self.y_old, self.y)


if __name__ == "__main__":
    # Example usage of the RK4 solver
    def fun(t, y):
        return -0.04 * y + 10 * np.exp(-((t - 4) ** 2) / 2)

    t0 = 0
    y0 = np.array([1])
    t_bound = 10

    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        fun,
        [t0, t_bound],
        y0,
        RK4,
    )

    print(sol.t)

    # solver = RK4(fun, t0, y0, t_bound)

    # while solver.status == "running":
    #     success, message = solver._step_impl()
    #     if not success:
    #         print(f"Solver Error: {message}")
    #         break

    #     print(f"Time: {solver.t}, State: {solver.y}")

    #     if solver.t >= t_bound:
    #         solver.status = "finished"

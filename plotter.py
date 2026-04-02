import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
sigma = 10
r = 28
b = 8/3

# Lorenz system
def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

# Time span
t_span = (0, 40)
t_eval = np.linspace(*t_span, 10000)

# Initial condition
state0 = [1.0, 1.0, 1.0]

# Solve ODE
sol = solve_ivp(lorenz, t_span, state0, t_eval=t_eval)

t = sol.t
x, y, z = sol.y

# --- Plot x(t), y(t), z(t) ---
fig, axs = plt.subplots(3, 1, sharex=True)

axs[0].plot(t, x)
axs[0].set_ylabel("x(t)")

axs[1].plot(t, y)
axs[1].set_ylabel("y(t)")

axs[2].plot(t, z)
axs[2].set_ylabel("z(t)")
axs[2].set_xlabel("time")

plt.tight_layout()
plt.show()

# --- Plot x vs z (phase plot) ---
plt.figure()
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("z")
plt.title("Lorenz Attractor (x vs z)")
plt.show()

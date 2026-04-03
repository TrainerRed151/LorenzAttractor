import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 10
r = 28
b = 8/3

# Time settings
dt = 0.01
t_max = 40
n_steps = int(t_max / dt)

# Allocate arrays
t = np.linspace(0, t_max, n_steps)
x = np.zeros(n_steps)
y = np.zeros(n_steps)
z = np.zeros(n_steps)

# Initial conditions
x[0], y[0], z[0] = 1.0, 1.0, 1.0

# Euler integration loop
for i in range(n_steps - 1):
    dx = sigma * (y[i] - x[i])
    dy = r * x[i] - y[i] - x[i] * z[i]
    dz = x[i] * y[i] - b * z[i]

    x[i+1] = x[i] + dt * dx
    y[i+1] = y[i] + dt * dy
    z[i+1] = z[i] + dt * dz

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

# --- Plot x vs z ---
plt.figure()
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("z")
plt.title("Lorenz Attractor (x vs z)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from generate_bile_salt_params import bile_salt_function, sum8_sin_func

# Load the fitted sine-series parameters for bile derivative
best_parameters = np.load("params/best_parameters.npy")

# ─── Model Parameters ───
mu_max   = 1.0    # max specific growth rate (1/h)
K        = 1.0    # carrying capacity (normalized)
epsilon  = 0.05   # coupling of substrate change to growth
Y        = 0.5    # biomass yield
theta1   = -1.0 / Y
theta2   = 0.1    # scaling of bile-derived substrate input
theta3   = -0.05  # bile impact on environmental HGT factor
theta4   = 0.325  # baseline environmental drift
theta5   = 0.1    # coupling constant for H factor
c        = 0.1    # metabolic cost of plasmid

# ─── Autonomous ODE System: [N, S, E, F, H, z] ───
def ode_system(z, y):
    N, S, E, F, H, dummy = y
    # Bile concentration and its derivative from fitted sine model
    B  = bile_salt_function(dummy)
    dB = sum8_sin_func(dummy, best_parameters)
    # Substrate change
    dS = theta1 * N + theta2 * dB
    # Population growth rate
    dN = mu_max * N * F * (1 - N/K) - epsilon * dS
    # Environmental stress factor
    dE = theta3 * B + theta4
    # Plasmid-free factor
    dF = H * (1 - E) * (1 - c)
    # Host/environment factor
    dH = theta5 * N * dS
    # Advance dummy time
    dz = 1.0
    return [dN, dS, dE, dF, dH, dz]

# ─── Initial Conditions and Simulation ───
y0    = [0.01, 2.1, 0.5, 0.9, 1.0, 0.0]  # N, S, E, F, H, z(0)
z_span = (0, 24)
z_eval = np.linspace(*z_span, 1000)

sol = solve_ivp(ode_system, z_span, y0, t_eval=z_eval,
                method="RK45", rtol=1e-6, atol=1e-9)

# ─── Extract Results ───
t_vals = sol.t
N_vals = sol.y[0]
F_vals = sol.y[3]
z_vals = sol.y[5]

# Compute growth rate dN/dt
dS_vals = theta1 * N_vals + theta2 * sum8_sin_func(z_vals, best_parameters)
growth_rate = mu_max * N_vals * F_vals * (1 - N_vals/K) - epsilon * dS_vals

# ─── Plot Growth Rate and Population Size ───
plt.figure(figsize=(10,4))
plt.plot(t_vals, growth_rate, label="Growth rate (dN/dt)", lw=2)
plt.xlabel("Time (h)")
plt.ylabel("Growth rate")
plt.title("Bacterial Growth Rate Over 24 h")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t_vals, N_vals / K, label="Population size (N/K)", color="tab:green", lw=2)
plt.xlabel("Time (h)")
plt.ylabel("Population size (fraction of K)")
plt.title("Bacterial Population Size Over 24 h")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from generate_bile_salt_params import param_sin_func

# Bile salt concentration model (smooth circadian pulses)
best_parameters = np.load("params/best_parameters.npy")
def sum8_sin_func(t, parameters=best_parameters):
    sin1 = param_sin_func(t, parameters[:3])
    sin2 = param_sin_func(t, parameters[3:6])
    sin3 = param_sin_func(t, parameters[6:9])
    sin4 = param_sin_func(t, parameters[9:12])
    sin5 = param_sin_func(t, parameters[12:15])
    sin6 = param_sin_func(t, parameters[15:18])
    sin7 = param_sin_func(t, parameters[18:21])
    sin8 = param_sin_func(t, parameters[21:])
    return sin1 + sin2 + sin3 + sin4 + sin5 + sin6 + sin7 + sin8


def bile_salt_derivative(y_dummy):
    """
    Instantaneous dB/dt at internal time y_dummy (hours).
    Wraps mod 24 for periodicity.
    """
    return sum8_sin_func(y_dummy % 24, best_parameters)

# Parameters
mu_max = 1.0    # 1/h, max growth rate
K = 1.0         # carrying capacity (normalized)
Y = 0.5       # yield coefficient (biomass per substrate) (0.45-0.6)
epsilon = 0.05  # coupling of substrate change to growth
theta1 = -1.0 / Y   # -2.0 with Y=0.5
theta2 = 0.14  # bile influence on substrate
theta3 = -0.05  # bile influence on HGT environment (negative = inhibitory)
theta4 = 0.325  # baseline environment drift
theta5 = 1e-4    # coupling for H factor
c = 0.1         # plasmid metabolic cost

# Initial conditions
N0 = 0.01; S0 = 1.1; E0 = 0.5; F0 = 0.9; H0 = 1.0
y0 = [N0, S0, E0, F0, H0, 1.4, 0.0]

# ODE system definition
def ode_system(t, y):
    N, S, E, F, H, B, y_dummy = y
    # Bile and its rate of change at time t
    dB_dt = bile_salt_derivative(y_dummy)
    # Substrate dynamics
    dS_dt = theta1 * N * S / (1 + S) * mu_max + theta2 * dB_dt  # consumption + bile-driven input
    # Bacterial population
    # dN_dt = mu_max * N * F * (1 - N/K) - epsilon * dS_dt
    dN_dt = mu_max * N * F * (S / (1 + S)) - epsilon * dS_dt
    # HGT environment factor
    dE_dt = theta3 * B  + theta4
    # Plasmid-free factor
    dF_dt = H * (1 - E) * (1 - c)
    # Host factor
    dH_dt = theta5 * N * dS_dt * E
    dy_dummy_dt = 1.0
    return [dN_dt, dS_dt, dE_dt, dF_dt, dH_dt, dB_dt, dy_dummy_dt]

# Integrate from t=0 to t=24 hours
t_span = (0, 24)
t_eval = np.linspace(0, 24, 1000)
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, dense_output=False)

# Plot all state variables in subplots
labels = ['N','S','E','F','H','B','y_dummy']
fig, axs = plt.subplots(len(labels), 1, figsize=(8, 2*len(labels)), sharex=True)
for i, label in enumerate(labels):
    axs[i].plot(sol.t, sol.y[i], label=label)
    axs[i].set_ylabel(label)
axs[-1].set_xlabel("Time (hours)")
fig.suptitle("State Variables Over 24 Hours")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

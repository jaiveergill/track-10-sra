import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from generate_bile_salt_params import param_sin_func

#bile salt concentration model
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

#defining our parameters
mu_max = 0.77  #in 1/h, max growth rate
K = 0.8         #carrying capacity (normalized)
Y = 0.4       #yield coefficient (biomass per substrate) (0.45-0.6)
epsilon = 0.05  #coupling of substrate change to growth
theta1 = -1.0 / Y   #-2.0 with Y=0.5
theta2 = 0.07  #bile influence on substrate
theta3 = -0.05  #bile influence on HGT environment (negative = inhibitory)


theta4 = 0.3

theta5 = 1    #coupling for H factor
c = 0.029       #plasmid metabolic cost

K_E   = 10.0

#defining our initial conditions
N0 = 0.01; S0 = 1.1; E0 = 0.5; F0 = 0.5; H0 = 0.2
y0 = [N0, S0, E0, F0, H0, 1.4, 0.0]

#ODE system definition

import numpy as np

def ode_system(t, y):
    N, S, E, F, H, B, y_dummy = y

    # 1) compute bile derivative as before
    dB_dt = bile_salt_derivative(y_dummy)
    monod = S / (1 + S)

    dS_dt = theta1 * N * monod * (1 - N / K) * mu_max + theta2 * dB_dt

    dN_dt = np.clip(mu_max * N * F * monod - epsilon * dS_dt, -N, None)

    dE_dt = np.clip(theta3 * B + theta4 + H, 0, 1) - E

    dF_dt = np.clip( H * (1 - E), 0, 1) - F

    dH_dt = ((1.0 - c) * monod - H / 2)

    dy_dummy_dt = 1.0
    return [dN_dt, dS_dt, dE_dt, dF_dt, dH_dt, dB_dt, dy_dummy_dt]


#integrating from t=0 to t=24 hours
t_span = (0, 24)
t_eval = np.linspace(0, 24, 1000)
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, dense_output=False)

#plot all state variables in subplots (this just for now to understand our dynamic variables)
labels = ['N','S','E','F','H','B','y_dummy']
fig, axs = plt.subplots(len(labels), 1, figsize=(8, 2*len(labels)), sharex=True)
for i, label in enumerate(labels):
    axs[i].plot(sol.t, sol.y[i], label=label)
    axs[i].set_ylabel(label)
axs[-1].set_xlabel("Time (hours)")
fig.suptitle("State Variables Over 24 Hours")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

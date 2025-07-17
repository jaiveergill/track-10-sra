import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from generate_bile_salt_params import param_sin_func

#BILE SALT MODEL
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

#PARAMETER DEFINITIONS
mu_max = 1.0    #in 1/h, max growth rate
K = 1.0         #carrying capacity (normalized)
Y = 0.5       #yield coefficient (biomass per substrate) (0.45-0.6)
epsilon = 0.05  #coupling of substrate change to growth
theta1 = -1.0 / Y   #-2.0 with Y=0.5
theta2 = 0.14  #bile influence on substrate
theta3 = -0.05  #bile influence on HGT environment (negative = inhibitory)
#theta4 = 0.325  #baseline environment drift, need to make a function of pH and temp 
theta5 = 1e-4    #coupling for H factor
c = 0.1         #plasmid metabolic cost

#INITIAL CONDITIONS
N0 = 0.01; S0 = 1.1; E0 = 0.5; F0 = 0.9; H0 = 1.0
y0 = [N0, S0, E0, F0, H0, 1.4, 0.0]

#THETA FOUR FUNCTION 
# physiological optima
PH_OPT       = 6.8
TEMP_OPT     = 37.0

#width of sensitivity
SIGMA_PH_DR  = 0.5   #pH units
SIGMA_TEMP_DR= 2.0   #°C

# peak baseline drift at the optimum
THETA4_MAX   = 0.325

def baseline_drift(pH, temp,
                   theta4_max=THETA4_MAX,
                   pH_opt=PH_OPT, sigma_pH=SIGMA_PH_DR,
                   temp_opt=TEMP_OPT, sigma_temp=SIGMA_TEMP_DR):
    """
    Returns a baseline environmental drift term (theta4) that is
    maximal (= theta4_max) at (pH_opt, temp_opt) and declines
    in a Gaussian manner as you move away.
    """
    ph_term   = np.exp(-((pH - pH_opt)**2)    / (2 * sigma_pH**2))
    temp_term = np.exp(-((temp - temp_opt)**2)/(2 * sigma_temp**2))
    return theta4_max * ph_term * temp_term

#testing conditions
ENV_pH   = 7.0   
ENV_temp = 38.5
theta4 = baseline_drift(ENV_pH, ENV_temp)

#ODE system definition
def ode_system(t, y):
    N, S, E, F, H, B, y_dummy = y
    #bile and its rate of change at time t
    dB_dt = bile_salt_derivative(y_dummy)
    #substrate dynamics
    dS_dt = theta1 * N * S / (1 + S) * mu_max + theta2 * dB_dt  #consumption + bile-driven input
    #growth rate 
    dN_dt = mu_max * N * F * (S / (1 + S)) - epsilon * dS_dt
    #HGT environment factor
    dE_dt = theta3 * dB_dt
    #plasmid-free factor
    dF_dt = H * (1 - E) * (1 - c) - F*0.15 #  linear decay term
    #host factor
    dH_dt = theta5 * N * dS_dt * E
    dy_dummy_dt = 1.0
    return [dN_dt, dS_dt, dE_dt, dF_dt, dH_dt, dB_dt, dy_dummy_dt]

#integrating from t=0 to t=24 hours
t_span = (0, 24)
t_eval = np.linspace(0, 24, 1000)
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, dense_output=False, method="BDF")

# Plot all state variables for the base simulation (for reference)
labels = ['N', 'S', 'E', 'F', 'H', 'B', 'y_dummy']
fig, axs = plt.subplots(len(labels), 1, figsize=(8, 2*len(labels)), sharex=True)
for i, label in enumerate(labels):
    axs[i].plot(sol.t, sol.y[i], label=label)
    axs[i].set_ylabel(label)
axs[-1].set_xlabel("Time (hours)")
fig.suptitle("Base Simulation: State Variables Over 24 Hours")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

theta4good = baseline_drift(6.8, 38.0)
theta4bad = baseline_drift(8.0, 39.5)

#SCENARIO FOR CHANGING PARAMETERS 
scenarios = {
    "Beneficial Plasmid": {
        "c": 0.1,
        "theta3": -0.05,
        "theta4": theta4good,
        "theta5": 1e-3
    },
    "Deleterious Plasmid": {
        "c": 0.1,
        "theta3": -0.05,
        "theta4": theta4bad,
        "theta5": 1e-3
    }
}

# Generating solutions for each scenario with modified initial condition for E0
solutions = {}
for name, params in scenarios.items():
    c = params["c"]
    theta3 = params["theta3"]
    theta4 = params["theta4"]
    theta5 = params["theta5"]
    # Adjust initial E value dynamically based on theta4
    y0_scenario = [N0, S0, (1 - theta4)*10, F0, H0, 1.4, 0.0]
    
    sol_scenario = solve_ivp(ode_system, t_span, y0_scenario, t_eval=t_eval,
                             method="RK45", rtol=1e-6, atol=1e-9)
    solutions[name] = sol_scenario

# Plot all state variables for each scenario using subplots
fig, axs = plt.subplots(len(labels), 1, figsize=(8, 2*len(labels)), sharex=True)
for i, label in enumerate(labels):
    for scenario, sol_scenario in solutions.items():
        axs[i].plot(sol_scenario.t, sol_scenario.y[i], label=scenario)
    axs[i].set_ylabel(label)
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Time (hours)")
plt.suptitle("State Variables for Beneficial vs. Deleterious Plasmid Scenarios")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
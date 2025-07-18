import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from generate_bile_salt_params import param_sin_func

# BILE SALT MODEL
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

mu_max = 0.3 # max growth rate, in 1/h,

epsilon = 0.05 # rate of population leaving with human meal times
 
Y = 0.5 # biomass yield coefficient
theta1 = -1.0 / Y # rate of substrate consumption
theta2 = 0.09 # bile influence on substrate influx
theta3 = -0.1  # bile influence on HGT environment (negative = inhibitory)
theta5 = 0.3 # environemental effects on HGT
c = 0.1 # plasmid metabolic cost


# INITIAL CONDITIONS
N0 = 0.1;
S0 = 1.1;
E0 = 0.5;
F0 = 0.9;
H0 = 0.5;
B0 = 1.4;
y0 = [N0, S0, E0, F0, H0, B0, 0.0] # 0.0 represent initial value of y_dummy 

# physiological optimum conditions
PH_OPT = 6.8
TEMP_OPT = 37.0
K_S = 1 # half-concentration constant

# width of sensitivity for the Gaussian
SIGMA_PH_DR  = 0.5   # pH units
SIGMA_TEMP_DR= 5.0   # deg C

# peak baseline environmental drift at the optimum conditions
THETA4_MAX   = 0.5

def baseline_drift(pH, temp,
                   theta4_max=THETA4_MAX,
                   pH_opt=PH_OPT, sigma_pH=SIGMA_PH_DR,
                   temp_opt=TEMP_OPT, sigma_temp=SIGMA_TEMP_DR):
    """
    Returns a baseline environmental drift term (theta4) that is
    maximum (theta4_max) at (pH_opt, temp_opt) and declines
    in a Gaussian manner as you move away from optimal conditions.
    """
    ph_term   = np.exp(-((pH - pH_opt)**2)    / (2 * sigma_pH**2))
    temp_term = np.exp(-((temp - temp_opt)**2)/(2 * sigma_temp**2))
    return 1-theta4_max * ph_term * temp_term # "1 -"  is done since a value farther from the optimum should be closer to 1 to scale environemental stress

# setting the base conditions at optimal environment
theta4 = baseline_drift(PH_OPT, TEMP_OPT)

#O DE SYSTEM DEFINITION
def ode_system(t, y, theta2_=theta2, theta3_=theta3, theta5_=theta5, c=c, theta4_val=theta4):
    """
    need big docstring here
    """
    N, S, E, F, H, B, y_dummy = y
    dB_dt = bile_salt_derivative(y_dummy)
    dS_dt = theta1 * N * S / (K_S + S) * mu_max + theta2_ * dB_dt
    dN_dt = mu_max * N * F * (S / (K_S + S)) - epsilon * dS_dt
    dE_dt = theta3_ * dB_dt * theta4_val
    dF_dt = H * (1 - E) * (1 - c) - F * 0.15
    dH_dt = theta5_ * N * S/(K_S + S)
    dy_dummy_dt = 1.0
    return [dN_dt, dS_dt, dE_dt, dF_dt, dH_dt, dB_dt, dy_dummy_dt]

days = 3
# integrating from t=0 to t=days*24 hours
t_span = (0, 24*days)
t_eval = np.linspace(0, 24*days, 1000)

if __name__ == "__main__":
    sol = solve_ivp(ode_system, t_span, y0, _eval=t_eval, dense_output=False, method="BDF")

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

    
    
# SCENARIOS FOR CHANGING PARAMETERS 

#generating different thetas, you can change the ph and temp

#changing temp
theta41 = baseline_drift(6.8, 36.0)
theta42 = baseline_drift(6.8, 36.4)
theta43 = baseline_drift(6.8, 36.8)
theta44 = baseline_drift(6.8, 37.2)
theta45 = baseline_drift(6.8, 37.6)
theta46 = baseline_drift(6.8, 38.0)
theta47 = baseline_drift(6.8, 38.4)


theta48 = baseline_drift(4.0, 37.0)

#changing pH
theta49 = baseline_drift(4.5, 37.0)
theta410 = baseline_drift(5.0, 37.0)
theta411 = baseline_drift(5.5, 37.0)
theta412 = baseline_drift(6.0, 37.0)
theta413 = baseline_drift(6.5, 37.0)
theta414 = baseline_drift(7.0, 37.0)

#constant
theta415 = baseline_drift(6.8, 37.0)


#connects to the thetas for theta4, and then you can change the c, theta3, and theta5 values
scenarios = {
    #changing temp
    "p1": {
        "c": c,
        "theta3": theta3,
        "theta4": theta41,
        "theta5": theta5
    },
    "p2": {
        "c": c,
        "theta3": theta3,
        "theta4": theta42,
        "theta5": theta5
    },
    "p3": {
       "c": c,
        "theta3": theta3,
        "theta4": theta43,
        "theta5": theta5
    },
    #changing pH
    "p4": {
        "c": c,
        "theta3": theta3,
        "theta4": theta44,
        "theta5": theta5
    },
    "p5": {
        "c": c,
        "theta3": theta3,
        "theta4": theta45,
        "theta5": theta5
    },
    "p6": {
        "c": c,
        "theta3": theta3,
        "theta4": theta46,
        "theta5": theta5
    },
    #changing c
    "p7": {
        "c": c-0.1,
        "theta3": theta3,
        "theta4": theta47,
        "theta5": theta5
    },
    "p8": {
        "c": c,
        "theta3": theta3,
        "theta4": theta48,
        "theta5": theta5
    },
    "p9": {
        "c": c+0.1,
        "theta3": theta3,
        "theta4": theta49,
        "theta5": theta5
    },
    #changing theta3
    "p10": {
        "c": c,
        "theta3": theta3,
        "theta4": theta410,
        "theta5": theta5
    },
     "p11": {
        "c": c,
        "theta3": theta3,
        "theta4": theta411,
        "theta5": theta5
    },
    "p12": {
        "c": c,
        "theta3": theta3,
        "theta4": theta412,
        "theta5": theta5
    },
    #changing theta5
     "p13": {
        "c": c,
        "theta3": theta3,
        "theta4": theta413,
        "theta5": theta5
    },
     "p14": {
        "c": c,
        "theta3": theta3,
        "theta4": theta414,
        "theta5": theta5
    },
     "p15": {
        "c": c,
        "theta3": theta3,
        "theta4": theta415,
        "theta5": theta5
    },
}

colormap = {
    #changing pH
    "p1":  "#deebf7",  # light blue
    "p2":  "#deebf7",  # light blue
    "p3":  "#9ecae1",  # medium blue

    #changing temp
    "p4":  "#9ecae1",  # medium blue
    "p5":  "#9ecae1",  # medium blue
    "p6":  "#3182bd",  # dark blue

    #changing c
    "p7":  "#3182bd",  # dark blue

    #base 
    "p8":  "#fcbba1",  # light red


    "p9":  "#fcbba1",  # light red

    #changing theta3
    "p10": "#fcbba1",  # light red
    "p11": "#fb6a4a",  # medium red
    "p12": "#fb6a4a",  # medium red

    #changing theta5
    "p13": "#cb181d",  # dark red
    "p14": "#cb181d",  # dark red
    "p15": "000000",  #black
}

opacity = {

    "p1":  0.5,
    "p2":  0.5,
    "p3":  0.5,
    "p4":  0.5,
    "p5":  0.5,
    "p6":  0.5,
    "p7":  0.5,
    "p8":  0.5,
    "p9":  0.5,
    "p10": 0.5,
    "p11": 0.5,
    "p12": 0.5,
    "p13": 0.5,
    "p14": 0.5,
    "p15": 1,
}

lw = {

    "p1":  2,
    "p2":  2,
    "p3":  2,
    "p4":  2,
    "p5":  2,
    "p6":  2,
    "p7":  2,
    "p8":  2,
    "p9":  2,
    "p10": 2,
    "p11": 2,
    "p12": 2,
    "p13": 2,
    "p14": 2,
    "p15": 0.5,
}


 # generating solutions for each scenario with modified initial condition for E0
solutions = {}
for name, params in scenarios.items():
    c = params["c"]
    theta3 = params["theta3"]
    theta4 = params["theta4"]
    theta5 = params["theta5"]
    # Adjust initial E value dynamically based on theta4
    y0_scenario = [N0, S0, (theta4), F0, H0, 1.4, 0.0]
    sol_scenario = solve_ivp(ode_system, t_span, y0_scenario, args=(theta2, theta3, theta5, c, theta4), t_eval=t_eval,
                            method="RK45", rtol=1e-6, atol=1e-9)
    solutions[name] = sol_scenario

#plot the growth rate for all scenarios 
plt.figure(figsize=(8, 4))
for name, sol in solutions.items():
    N_vals = sol.y[0]
    S_vals = sol.y[1]
    dB_vals = bile_salt_derivative(sol.y[6])
    dS_vals = theta1 * N_vals * S_vals / (1 + S_vals) * mu_max + theta2 * dB_vals
    growth_rate = mu_max * N_vals * sol.y[3] * (S_vals / (1 + S_vals)) - epsilon * dS_vals
    plt.plot(sol.t, growth_rate, color=colormap[name], alpha=opacity[name], linewidth=lw[name], label=name)
plt.title("Growth Rate Across Scenarios")
plt.xlabel("Time (hours)")
plt.ylabel("dN/dt")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#plotting N(t) for all plasmid scenarios 
plt.figure(figsize=(8, 4))
for name, sol in solutions.items():
    plt.plot(sol.t, sol.y[0], label=name, color=colormap[name], alpha=opacity[name], linewidth=lw[name])
plt.title("Population Size N(t) Across Plasmid Scenarios")
plt.xlabel("Time (hours)") 
plt.ylabel("Population N")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#plot all state variables for each scenario using subplots
fig, axs = plt.subplots(len(labels), 1, figsize=(8, 2*len(labels)), sharex=True)
for i, label in enumerate(labels): 
    for name, sol_scenario in solutions.items():
        axs[i].plot(sol_scenario.t, sol_scenario.y[i], color=colormap[name], alpha=opacity[name], linewidth=lw[name], label=name)
    axs[i].set_ylabel(label)
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Time (hours)")
plt.suptitle("State Variables for Plasmid Scenarios")
plt.show()


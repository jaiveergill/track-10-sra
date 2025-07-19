import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from generate_bile_salt_params import param_sin_func
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.cm       as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, FuncFormatter


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
theta5 = 0.01 # environemental effects on HGT
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
PH_OPT = 6.5
TEMP_OPT = 37.0
K_S = 1 # half-concentration constant

# width of sensitivity for the Gaussian
SIGMA_PH_DR  = 0.5   # pH units
SIGMA_TEMP_DR= 4.0   # deg C
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

#setting the base conditions at optimal environment
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

days = 1
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

#changing temp
theta41 = baseline_drift(6.5, 33.0)
theta42 = baseline_drift(6.5, 34.0)
theta43 = baseline_drift(6.5, 35.0)
theta44 = baseline_drift(6.5, 36.0)
theta45 = baseline_drift(6.5, 37.0)
theta46 = baseline_drift(6.5, 38.0)
theta47 = baseline_drift(6.5, 39.0)

#changing pH
theta48 = baseline_drift(4.0, 37.0)
theta49 = baseline_drift(4.75, 37.0)
theta410 = baseline_drift(5.5, 37.0)
theta411 = baseline_drift(6.25, 37.0)
theta412 = baseline_drift(7.0, 37.0)
theta413 = baseline_drift(7.75, 37.0)
theta414 = baseline_drift(8.5, 37.0)

#constant
theta415 = baseline_drift(6.5, 37.0)


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
    "p7": {
        "c": c-0.1,
        "theta3": theta3,
        "theta4": theta47,
        "theta5": theta5
    },
    #changing pH
    "p8": {
        "c": c,
        "theta3": theta3,
        "theta4": theta48,
        "theta5": theta5
    },
    "p9": {
        "c": c,
        "theta3": theta3,
        "theta4": theta49,
        "theta5": theta5
    },
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
    #baseline 
     "p15": {
        "c": c,
        "theta3": theta3,
        "theta4": theta415,
        "theta5": theta5
    },
}

colormap = {
    #changing temp
    "p1":  "#B6E7FF",  # light blue
    "p2":  "#9ADEFF",  # light blue
    "p3":  "#84D7FF",  # medium blue
    "p4":  "#66CDFF",  # medium blue
    "p5":  "#4EC5FF",  # medium blue
    "p6":  "#2BB9FF",  # dark blue
    "p7":  "#00ABFF",  # dark blue

    #changing pH
    "p8":  "#FFD4D4",  # light red
    "p9":  "#FFBDBD",  # light red
    "p10": "#FF9F9F",  # light red
    "p11": "#FF7A7A",  # medium red
    "p12": "#FF5E5E",  # medium red
    "p13": "#FF4141",  # dark red
    "p14": "#FF2929",  # dark red
    "p15": "000000",   # black
}

opacity = {

    "p1":  0.75,
    "p2":  0.75,
    "p3":  0.75,
    "p4":  0.75,
    "p5":  0.75,
    "p6":  0.75,
    "p7":  0.75,
    "p8":  0.75,
    "p9":  0.75,
    "p10": 0.75,
    "p11": 0.75,
    "p12": 0.75,
    "p13": 0.75,
    "p14": 0.75,
    "p15": 1,
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
fig, ax = plt.subplots(figsize=(8,4))
for name, sol in solutions.items():
    N_vals = sol.y[0]
    S_vals = sol.y[1]
    dB_vals = bile_salt_derivative(sol.y[6])
    dS_vals = theta1 * N_vals * S_vals / (1 + S_vals) * mu_max + theta2 * dB_vals
    growth_rate = mu_max * N_vals * sol.y[3] * (S_vals / (1 + S_vals)) - epsilon * dS_vals
    plt.plot(sol.t, growth_rate, color=colormap[name], alpha=opacity[name], linewidth=1, label=name)

#colorbar for blue
norm_temp = mcolors.Normalize(vmin=33, vmax=38)
sm_temp   = cm.ScalarMappable(norm=norm_temp, cmap="Blues")
sm_temp.set_array([])

cax_temp = inset_axes(ax,
                      width="20%",     # 30% of parent axes width
                      height="3%",     # 3% of parent axes height
                      loc="lower left",
                      bbox_to_anchor=(0.04, 0.08, 1, 1),
                      bbox_transform=ax.transAxes)
cbar_temp = fig.colorbar(sm_temp, cax=cax_temp, orientation="horizontal")
cbar_temp.set_ticks([33, 38])
cbar_temp.ax.tick_params(labelsize=9)
cbar_temp.set_label("Temp (°C)", labelpad=-11, fontsize=9)

#colorbar for red
norm_ph = mcolors.Normalize(vmin=4.5, vmax=8.5)
sm_ph   = cm.ScalarMappable(norm=norm_ph, cmap="Reds")
sm_ph.set_array([])

cax_ph = inset_axes(ax,
                    width="20%",
                    height="3%",
                    loc="lower left",
                    bbox_to_anchor=(0.04, 0.18, 1, 1),
                    bbox_transform=ax.transAxes)
cbar_ph = fig.colorbar(sm_ph, cax=cax_ph, orientation="horizontal")
cbar_ph.set_ticks([4.5, 8.5],)
cbar_ph.ax.tick_params(labelsize=9)
cbar_ph.set_label("pH level", labelpad=-12, fontsize=9)

# 4) add a single black‐line legend entry, above those bars
baseline_handle = Line2D([0],[0], color="k", lw=2)
legend_ax = inset_axes(ax,
                       width="30%",
                       height="6%",
                       loc="lower left",
                       bbox_to_anchor=(0.029, 0.23, 1, 1),
                       bbox_transform=ax.transAxes)
legend_ax.axis("off")
legend_ax.legend([baseline_handle],
                 ["Baseline state"],
                 loc="center left",
                 frameon=False,
                 fontsize=9)
ax.set_title("Growth Rate Varying pH and Temp")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("dN/dt")
ax.grid(True)
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.grid(which='minor', linestyle='-', linewidth=0.8, color='gray', alpha=0.7)
plt.sca(ax)
plt.xlim(0,24*days)
plt.ylim(-0.6,0.6)
plt.show()

#plotting N(t) for all plasmid scenarios 
plt.figure(figsize=(8, 4))
for name, sol in solutions.items():
    plt.plot(sol.t, sol.y[0], label=name, color=colormap[name], alpha=opacity[name], linewidth=1)
plt.title("Population Size N(t) Across Plasmid Scenarios")
plt.xlabel("Time (hours)") 
plt.ylabel("Population N")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0,24*days)
plt.show()

#Plotting fitness variable F 
plt.figure(figsize=(8, 4))
for name, sol in solutions.items():
    F_vals = sol.y[3]   # index 3 is the fitness variable F
    plt.plot(sol.t, F_vals, color=colormap[name], alpha=opacity[name], linewidth=1, label=name)
plt.title("Fitness F(t) Across Scenarios")
plt.xlabel("Time (hours)")
plt.ylabel("Fitness F")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(0,24*days)
plt.show()

#plot all state variables for each scenario using subplotspy
fig, axs = plt.subplots(len(labels), 1, figsize=(8, 2*len(labels)), sharex=True)
for i, label in enumerate(labels): 
    for name, sol_scenario in solutions.items():
        axs[i].plot(sol_scenario.t, sol_scenario.y[i], color=colormap[name], alpha=opacity[name], linewidth=1, label=name)
    axs[i].set_ylabel(label)
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Time (hours)")
plt.suptitle("State Variables for Plasmid Scenarios")
plt.xlim(0,24*days)
plt.show()


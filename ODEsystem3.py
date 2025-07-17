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

#max growth rate, in 1/h,
mu_max = 0.3    
#carrying capacity (normalized)
K = 1.0    
#yield coefficient (biomass per substrate) (0.45-0.6)     
Y = 0.5       
#coupling of substrate change to growth
epsilon = 0.05 
#-2.0 with Y=0.5
theta1 = -1.0 / Y  
#bile influence on substrate
theta2 = 0.09

#bile influence on HGT environment (negative = inhibitory)
theta3 = -0.1  
#coupling for H factor
theta5 = 1e-4
#plasmid metabolic cost
c = 0.02    

#INITIAL CONDITIONS

N0 = 0.1; S0 = 1.1; E0 = 0.5; F0 = 0.9; H0 = 0.5
y0 = [N0, S0, E0, F0, H0, 1.4, 0.0]

#THETA FOUR FUNCTION 

# physiological optima
PH_OPT       = 6.8
TEMP_OPT     = 37.0
K_S = 1

#width of sensitivity
SIGMA_PH_DR  = 0.5   #pH units
SIGMA_TEMP_DR= 5.0   #°C

#peak baseline environmental drift at the optimum
THETA4_MAX   = 0.5

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
    return 1-theta4_max * ph_term * temp_term

#setting the base conditions at optimal environment
theta4 = baseline_drift(PH_OPT, TEMP_OPT)

#ODE SYSTEM DEFINITION
def ode_system(t, y, theta2_=theta2, theta3_=theta3, theta5_=theta5, c_=c, theta4_val=None):
    if theta4_val is None:
        theta4_val = theta4
    N, S, E, F, H, B, y_dummy = y
    dB_dt = bile_salt_derivative(y_dummy)
    # substrate dynamics with custom theta2
    dS_dt = theta1 * N * S / (K_S + S) * mu_max + theta2_ * dB_dt
    dN_dt = mu_max * N * F * (S / (K_S + S)) - epsilon * dS_dt
    # HGT environment factor with custom theta3 and theta4_val
    dE_dt = theta3_ * dB_dt * theta4_val
    # plasmid-free factor using custom cost c_
    dF_dt = H * (1 - E) * (1 - c_) - F * 0.15
    dH_dt = theta5_ * N * S/(K_S + S)
    dy_dummy_dt = 1.0
    return [dN_dt, dS_dt, dE_dt, dF_dt, dH_dt, dB_dt, dy_dummy_dt]



t_span = (0, 48)
t_eval = np.linspace(0, 48, 1000)
if __name__ == "__main__":
    #integrating from t=0 to t=48 hours
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

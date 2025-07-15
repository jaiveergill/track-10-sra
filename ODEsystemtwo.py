import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Bile salt concentration model (smooth circadian pulses)
def logistic(x, k=10):
    return 1 / (1 + np.exp(-k * x))

def box_function(t, a, b, k=10):
    """Smooth pulse from time a to b using sigmoids."""
    return logistic(t - a, k) - logistic(t - b, k)

def bile_salt_function(t):
    # Baseline bile level and three meal-induced pulses
    base = 4.0  # baseline (μM, for example)
    pulses = [(9, 10), (13, 14), (19, 20)]  # pulse intervals after meals
    value = base
    for (start, end) in pulses:
        # Add a high peak (12 units) in first hour and a smaller tail (8 units) in second hour
        value += 12 * box_function(t, start, start+1) 
        value += 8  * box_function(t, start+1, end)
    return value

def bile_salt_derivative(t):
    # Analytical derivative of the bile salt function
    k = 10
    db_dt = 0.0
    pulses = [(9, 10), (13, 14), (19, 20)]
    for (start, end) in pulses:
        # derivative of logistic part for rising edge (start) and falling edge (end)
        sig1 = logistic(t - start, k);  sig2 = logistic(t - (start+1), k)
        d_box1 = k * (sig1 * (1 - sig1) - sig2 * (1 - sig2))
        sig3 = logistic(t - (start+1), k);  sig4 = logistic(t - end, k)
        d_box2 = k * (sig3 * (1 - sig3) - sig4 * (1 - sig4))
        db_dt += 12 * d_box1 + 8 * d_box2
    return db_dt

# Parameters
mu_max = 1.0    # 1/h, max growth rate
K = 1.0         # carrying capacity (normalized)
Y = 0.5         # yield coefficient (biomass per substrate)
epsilon = 0.05  # coupling of substrate change to growth
theta1 = -1.0 / Y   # -2.0 with Y=0.5
theta2 = 0.1    # bile influence on substrate
theta3 = -0.05  # bile influence on HGT environment (negative = inhibitory)
theta4 = 0.325  # baseline environment drift
theta5 = 0.1    # coupling for H factor
c = 0.1         # plasmid metabolic cost

# Initial conditions
N0 = 0.01; S0 = 2.1; E0 = 0.5; F0 = 0.9; H0 = 1.0
y0 = [N0, S0, E0, F0, H0]

# ODE system definition
def ode_system(t, y):
    N, S, E, F, H = y
    # Bile and its rate of change at time t
    B = bile_salt_function(t)
    dB_dt = bile_salt_derivative(t)
    # Substrate dynamics
    dS_dt = theta1 * N + theta2 * dB_dt   # consumption + bile-driven input
    # Bacterial population
    dN_dt = mu_max * N * F * (1 - N/K) - epsilon * dS_dt
    # HGT environment factor
    dE_dt = theta3 * B + theta4
    # Plasmid-free factor
    dF_dt = H * (1 - E) * (1 - c)
    # Host factor
    dH_dt = theta5 * N * dS_dt
    return [dN_dt, dS_dt, dE_dt, dF_dt, dH_dt]

# Integrate from t=0 to t=24 hours
t_span = (0, 24)
t_eval = np.linspace(0, 24, 1000)
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, dense_output=False)

# Extract solution for N over time
N_t = sol.y[0]

# Plot the bacterial growth curve
plt.figure(figsize=(6,4))
plt.plot(sol.t, N_t, label="Bacterial population N(t)", color='orange')
plt.axhline(y=1.0, color='gray', linestyle='--', label="Carrying capacity K")
plt.xlabel("Time (hours)")
plt.ylabel("Bacterial Population (fraction of carrying capacity)")
plt.title("Bacterial Growth Over 24 Hours")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

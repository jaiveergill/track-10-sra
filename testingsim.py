import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import product
from generate_bile_salt_params import param_sin_func

# ─── Bile‐salt derivative via 8‐term sine series ───
best_parameters = np.load("params/best_parameters.npy")
def sum8_sin_func(t, parameters=best_parameters):
    return sum(param_sin_func(t, parameters[3*i:3*i+3]) for i in range(8))

def bile_salt_derivative(z):
    return sum8_sin_func(z % 24)

# ─── Fixed Model Parameters ───
mu_max   = 0.3
K        = 1.0
Y        = 0.5
epsilon  = 0.05
theta1   = -1.0 / Y
theta2   = 0.14

# ─── baseline_drift(pH, T) ───
PH_OPT, TEMP_OPT = 6.8, 37.0
SIGMA_PH, SIGMA_TEMP = 0.5, 5.0
THETA4_MAX = 0.5

def baseline_drift(pH, temp):
    ph_term   = np.exp(-((pH - PH_OPT)**2)    / (2 * SIGMA_PH**2))
    temp_term = np.exp(-((temp - TEMP_OPT)**2)/(2 * SIGMA_TEMP**2))
    return THETA4_MAX * ph_term * temp_term

# ─── ODE system ───
def ode_system(t, y, c, theta3, theta5, theta4):
    N, S, E, F, H, B, z = y
    dB  = bile_salt_derivative(z)
    dS  = theta1 * N * S/(1+S) * mu_max + theta2 * dB
    dN  = mu_max * N * F * (S/(1+S)) - epsilon * dS
    dE  = theta3 * dB + theta4
    dF  = H * (1 - E) * (1 - c)
    dH  = theta5 * N * dS * E
    dz  = 1.0
    return [dN, dS, dE, dF, dH, dB, dz]

# ─── Parameter Ranges ───
pH_values     = [6.5, 6.8, 7.2]       # example pH range
temp_values   = [37.0, 39.0]          # example temperature range
c_values      = [0.0, 0.1]            # plasmid neutral vs. cost
theta3_values = [-0.05, -0.1]         # weak vs. strong bile inhibition
theta5_values = [1e-4, 5e-4]          # slow vs. fast H coupling

# ─── Simulation Setup ───
t_span = (0, 48)
t_eval = np.linspace(0, 48, 1000)
y0 = [0.1, 1.1, 0.5, 0.9, 1.0, bile_salt_derivative(0), 0.0]

# ─── Run sweep ───
results = {}
for (pH, temp, c, th3, th5) in product(pH_values, temp_values, c_values, theta3_values, theta5_values):
    th4 = baseline_drift(pH, temp)
    key = f"pH{pH}_T{temp}_c{c}_θ3{th3}_θ5{th5}"
    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, c, th3, th5, th4),
        t_span=t_span, y0=y0, t_eval=t_eval,
        rtol=1e-6, atol=1e-9
    )
    results[key] = sol

# ─── Plot N(t) ───
plt.figure(figsize=(8,4))
for key, sol in results.items():
    plt.plot(sol.t, sol.y[0], label=key)
plt.title("Population N(t) Across Parameter Combinations")
plt.xlabel("Time (h)")
plt.ylabel("N")
plt.legend(fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── Plot Growth Rate dN/dt ───
plt.figure(figsize=(8,4))
for key, sol in results.items():
    N, S, F, z = sol.y[0], sol.y[1], sol.y[3], sol.y[6]
    dB = bile_salt_derivative(z)
    dS = theta1 * N * S/(1+S) * mu_max + theta2 * dB
    gr = mu_max * N * F * (S/(1+S)) - epsilon * dS
    plt.plot(sol.t, gr, label=key)
plt.title("Growth Rate dN/dt Across Parameter Combinations")
plt.xlabel("Time (h)")
plt.ylabel("dN/dt")
plt.legend(fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── Plot Fitness F(t) ───
plt.figure(figsize=(8,4))
for key, sol in results.items():
    plt.plot(sol.t, sol.y[3], label=key)
plt.title("Fitness F(t) Across Parameter Combinations")
plt.xlabel("Time (h)")
plt.ylabel("F")
plt.legend(fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

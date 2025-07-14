# DON'T RUN THIS YET IM STILL WORKING ON IT PLS !! :)
# Numpy, Scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint

#Imports
from odesystem import (
    mu, dN_dt, dS_dt, dF_dt, dE_dt, H,
    Z_pH, Z_temp, Z_bile,
    dD_dt, dR_dt, bile_salt_function_differential_equation
)
from constants import *
from generate_bile_salt_params import sum8_sin_func

# Creating the time grid
t_values = np.linspace(TIME_START, TIME_END, 1000)

#Setting the initial conditions
initial_conditions = [INITIAL_N, INITIAL_D, INITIAL_R, INITIAL_S, INITIAL_F, INITIAL_E, 0]  # last one is dummy bile timer y=0

# Basic system framework
def ecoli_system(t, y):
    N, D, R, S, F, E, y_dummy = y
    # Compute bile(t), pH(t), T(t)
    # Compute Z_* terms
    # Compute H(t), mu(t), and all d/dt values
    return [dN_dt, dD_dt, dR_dt, dS_dt, dF_dt, dE_dt, dy_dummy]

# Solving the system using solve_ivp
solution = solve_ivp(ecoli_system, [TIME_START, TIME_END], initial_conditions, t_eval=t_values)

# Plot results 
'''
N(t), D(t), R(t)

F(t) and E(t)

S(t) and maybe H(t) (recompute H at each t if needed)
'''
t = solution.t
N_vals, D_vals, R_vals, S_vals, F_vals, E_vals, dummy_vals = solution.y
plt.figure(figsize=(10, 6))
plt.plot(t, N_vals, label="Total Population (N)")
plt.plot(t, D_vals, label="Donor Cells (D)")
plt.plot(t, R_vals, label="Recipient Cells (R)")
plt.xlabel("Time (hours)")
plt.ylabel("Cell Count")
plt.title("Bacterial Population Dynamics")
plt.legend()
plt.grid(True)
plt.show()


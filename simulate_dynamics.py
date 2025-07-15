# DON'T RUN THIS YET IM STILL WORKING ON IT PLS !! :)
# Numpy, Scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

best_parameters = np.load("params/best_parameters.npy")

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
    '''
    * Receives current time t and state vector y
    * Unpacks all current state values: N, D, R, S, F, E, y_dummy
    * Computes:
        * bile(t) using sine model
        * environmental penalties Z_ph, Z_temp, Z_bile
        * environmental stress rate dE/dt
        * HGT rate (H(t))
        * Growth rate mu(t)
        * Population change dN/dt, substrate use dS/dt, etc.
    * Returns a list of all derivatives in the same order

    Steps so I don't lose my mind:
    1. Unpack y: get current values of -> N, D, R, S, F, E, y_dummy = y
    2. Define environmental inputs (can be constants for now): pH = PH_OPT, temp = TEMP_OPT, bile(t) = sum8_sin_func(y_dummy, best_parameters)
    3. Compute Z-values (Z_ph, Z_temp, Z_bile)
    4. Compute dE/dt
    5. Compute HGT rate H(t)
    6. Compute dF/dt
    7. Compute mu(t)
    8. Compute dN/dt
    9. Compute dS/dt
    10. Compute dD/dt and dR/dt
    11. Compute dy_dummy/dt = 1
    12. Return the derivative vector: return [dN_dt_val, dD_dt_val, dR_dt_val, dS_dt_val, dF_dt_val, dE_dt_val, 1.0]
    '''


    N, D, R, S, F, E, y_dummy = y
    
    if abs(t) < 1e-6:
        print("Initial bile(t):", sum8_sin_func(y_dummy, best_parameters))

    print(f"t = {t:.2f}")
    # if t > 0.2:
    #     raise ValueError("Aborting early to prevent infinite loop")

  

    #2
    pH = PH_OPT
    temp = TEMP_OPT
    bile = sum8_sin_func(y_dummy, best_parameters)
    if abs(t) < 1e-6:
        print(f"Initial bile concentration: {bile:.4f}")


    #3
    Z_pH_val = Z_pH(pH, PH_OPT, SIGMA_PH)
    Z_temp_val = Z_temp(temp, TEMP_OPT, SIGMA_TEMP)
    Z_bile_val = Z_bile(bile, BILE_OPT, SIGMA_BILE)

    #4
    dE_dt_val = dE_dt(THETA_PH, Z_pH_val, THETA_TEMP, Z_temp_val, THETA_BILE, Z_bile_val)
    # E = max(0.0, min(E, 0.99))
    E += dE_dt_val
    E = np.clip(E, 0.0, 0.99)  # Constrain E to [0, 0.99]


    #5
    H_val = H(t, BETA_MAX, D, R, S, K_S, E)

    #6
    dF_dt_val = dF_dt(H_val, E, CP)

    # #7
    # #mu(t, mu_max, F, N, K, epsilon, dS_dt)
    # mu_val = mu(t, MU_MAX, F, N, CARRYING_CAPACITY, EPSILON, dS_dt_val) #dE/dt placeholder for dS/dt

    # #8
    # dN_dt_val = dN_dt(mu_val, N)

    # #9
    mu_temp = 0.7* MU_MAX
    dN_dt_val = dN_dt(mu_temp, N)
    dS_dt_val = dS_dt(dN_dt_val, YIELD,S_IN)
    mu_val = mu(t, MU_MAX, F, N, CARRYING_CAPACITY, EPSILON, dS_dt_val)
    

    if t < 0.1:
        print(f"t={t:.2f}, bile={bile:.2f}, E={E:.2f}, H={H_val:.2e}, mu={mu_val:.4f}, dN/dt={dN_dt_val:.2f}")

    # 7 & 8: compute dN_dt (temporarily use dummy mu)
      # or a safe guess

    # 9: compute dS_dt based on dN_dt
    # dS_dt_val = dS_dt(dN_dt_val, YIELD, S_IN)

    # Now that we have dS/dt, recompute actual mu
    

    # Update dN/dt using correct mu
    dN_dt_val = dN_dt(mu_val, N)

    
    #10
    dD_dt_val = dD_dt(dN_dt_val, D, N, H_val, CP)
    dR_dt_val = dR_dt(dN_dt_val, R, N, H_val)

    #11
    dy_dummy_dt = 1.0
   
    #12
    return [dN_dt_val, dD_dt_val, dR_dt_val, dS_dt_val, dF_dt_val, dE_dt_val, dy_dummy_dt]



# Solving the system using solve_ivp
# solution = solve_ivp(
#     ecoli_system,
#     [TIME_START, TIME_END],
#     [INITIAL_N, INITIAL_D, INITIAL_R, INITIAL_S, INITIAL_F, INITIAL_E, 0],  # initial state
#     t_eval=np.linspace(TIME_START, TIME_END, 1000)
# )
solution = solve_ivp(
    ecoli_system,
    [TIME_START, TIME_END],
    initial_conditions,
    t_eval=np.linspace(TIME_START, TIME_END, 1000),
    method="BDF",  # <—— STIFF SOLVER
    rtol=1e-5,
    atol=1e-8
)

print("Any NaNs?", np.isnan(solution.y).any())
print("Any Infs?", np.isinf(solution.y).any())
print("Solution success:", solution.success)
print("Message:", solution.message)



# Plot results 
'''
N(t), D(t), R(t)

F(t) and E(t)

S(t) and maybe H(t) (recompute H at each t if needed)
'''
t = solution.t
dN_dt_vals, dD_dt_vals, dR_dt_vals, dS_dt_vals, dF_dt_vals, dE_dt_vals, dy_dummy_dt = solution.y

plt.figure(figsize=(10, 6))
plt.plot(t, dN_dt_vals, label="Total Population (N)")
plt.plot(t, dD_dt_vals, label="Donor Cells (D)")
plt.plot(t, dR_dt_vals, label="Recipient Cells (R)")
plt.xlabel("Time (hours)")
plt.ylabel("Cell Count")
plt.title("Bacterial Population Dynamics")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, dF_dt_vals, label="Fitness (F)")
plt.plot(t, dE_dt_vals, label="Environmental Stress (E)")
plt.plot(t, dS_dt_vals, label="Substrate (S)")
plt.legend()
plt.title("Fitness, Stress, and Substrate over Time")
plt.grid(True)
plt.show()

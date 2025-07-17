from ODEsystem3 import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def run_grid_search(param_ranges):
    F0_const = 0.9
    N0_, S0_, H0_ = N0, S0, H0
    other = [1.4, 0.0]
    c_vals    = param_ranges.get('c', [0.05, 0.15])
    pH_vals   = param_ranges.get('pH', [6.5, 6.9])
    temp_vals = param_ranges.get('temp', [36.0, 38.0])
    theta2_vals = param_ranges.get('theta2', [0.05, 0.12])
    theta3_vals = param_ranges.get('theta3', [-0.15, -0.05])
    theta5_vals = param_ranges.get('theta5', [5e-5, 1.5e-4])
    
    scenarios = {}
    count = 0
    for (c_val, pH_val, temp_val, theta2_val, theta3_val, theta5_val) in product(c_vals, pH_vals, temp_vals, theta2_vals, theta3_vals, theta5_vals):
        count += 1
        E0_val = baseline_drift(pH_val, temp_val)
        F0_val = F0_const
        y0 = [N0_, S0_, E0_val, F0_val, H0_] + other
        theta4_val = baseline_drift(pH_val, temp_val)
        ode_func = lambda t, y: ode_system(t, y, theta2_=theta2_val, theta3_=theta3_val, theta5_=theta5_val, c_=c_val, theta4_val=theta4_val)
        sol = solve_ivp(ode_func, t_span, y0, t_eval=t_eval, method="BDF", rtol=1e-6, atol=1e-9)
        scenarios[f"scenario_{count}"] = {"solution": sol, "parameters": {"c": c_val, "pH": pH_val, "temp": temp_val, "theta2": theta2_val, "theta3": theta3_val, "theta5": theta5_val, "theta4": theta4_val, "E0": E0_val, "F0": F0_val}}
    print("Total scenarios run:", count)
    return scenarios

def graph_outputs(scenarios):
    params_to_plot = ['c', 'pH', 'temp', 'theta2', 'theta3', 'theta5']
    num_plots = len(params_to_plot)
    fig, axs = plt.subplots(num_plots, 1, figsize=(6, 4 * num_plots))
    if num_plots == 1:
        axs = [axs]
    for i, param in enumerate(params_to_plot):
        x_data = []
        y_data = []
        for key, scenario in scenarios.items():
            param_value = scenario['parameters'][param]
            sol = scenario['solution']
            final_N = sol.y[0, -1]
            x_data.append(param_value)
            y_data.append(final_N)
        axs[i].scatter(x_data, y_data)
        axs[i].set_xlabel(param)
        axs[i].set_ylabel("Final N")
        axs[i].set_title(f"Effect of {param} on Final N")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    param_ranges = {
        'c': [0.05, 0.10, 0.15],
        'pH': [6.5, 6.7, 6.9],
        'temp': [36.0, 37.0, 38.0],
        'theta2': [0.05, 0.085, 0.12],
        'theta3': [-0.15, -0.10, -0.05],
        'theta5': [5e-5, 1e-4, 1.5e-4]
    }
    scenarios = run_grid_search(param_ranges)

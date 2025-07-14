from scipy.optimize import minimize
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def box(t, a, b, k=10):
    """Smooth box function from a to b using sigmoids."""
    return 1 / (1 + np.exp(-k * (t - a))) - 1 / (1 + np.exp(-k * (t - b)))

def bile_salt_function(t, k=10):
    base = 4
    peaks = [9, 13, 19]
    value = base
    for h in peaks:
        value += 12 * box(t, h, h+1, k)
        value += 8 * box(t, h+1, h+2, k)
    return value

def param_sin_func(t, parameters):
    freq_shift, freq, scale = parameters
    return scale * np.sin(freq * t + freq_shift)

def metric_of_success(parameters):
    return np.mean((param_sin_func(time, parameters) - signal)**2)

def sum8_sin_func(t, parameters):
    sin1 = param_sin_func(t, parameters[:3])
    sin2 = param_sin_func(t, parameters[3:6])
    sin3 = param_sin_func(t, parameters[6:9])
    sin4 = param_sin_func(t, parameters[9:12])
    sin5 = param_sin_func(t, parameters[12:15])
    sin6 = param_sin_func(t, parameters[15:18])
    sin7 = param_sin_func(t, parameters[18:21])
    sin8 = param_sin_func(t, parameters[21:])
    return sin1 + sin2 + sin3 + sin4 + sin5 + sin6 + sin7 + sin8

def metric_of_success(parameters):
    return np.mean((sum8_sin_func(time, parameters) - signal)**2)

# Compare to the original bile concentration profile
def bile_salt_differential_equation(x, t):
    """ Differential equation describing the nonlinear oscillation of bile concentration over a 24 hour cycle.
    @param x: a 1D array of length 2 of floats, the first entry is the molar bile concentration and the second
        entry is a dummy variable that keeps track of time. So at t=0, the second entry should be 0.
    @param t: nonegative float, the time at which to evaluate the differential equation.
    @returns a 1D array of length 2 of floats, the time derivative of the bile concentration and the dummy variable.
    """
    bile, y = x #bile2 and bile1
    bile = sum8_sin_func(y, best_parameters) # Our model for the dynamics
    return np.array([bile, 1])


if __name__ == "__main__":
    t_values = np.linspace(0, 24, 1000)
    bile_values = [bile_salt_function(t) for t in t_values]

    time = t_values
    signal = np.gradient(bile_values, t_values)

    best_metric_of_success = np.inf
    best_parameters = None
    for i in range(20):
        results = minimize(metric_of_success, np.random.rand(24))
        if results.fun < best_metric_of_success:
            best_metric_of_success = results.fun
            best_parameters = results.x

    print(best_parameters)
    np.save("params/best_parameters.npy", best_parameters)
        

    simulated_bile = odeint(bile_salt_differential_equation, [4, 0], t_values)
    plt.plot(t_values, bile_values, label="Original Bile Concentration")
    plt.plot(t_values, simulated_bile[:, 0], label="Simulated Bile Concentration")
    plt.xlabel("Time (h)")
    plt.ylabel("Bile concentration")
    plt.title("Time Series Signal with Simulated Bile Concentration")
    plt.legend()
    plt.show()
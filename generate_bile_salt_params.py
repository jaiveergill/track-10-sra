"""
This script generates the parameters for the bile salt function.
"""

# imports
from scipy.optimize import minimize
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) #To introduce a level of randomness

def box(t, a, b, k=10):
    """
    Smooth step function approximating a box from time `a` to `b`. Uses the difference of two sigmoids to simulate a smooth rectangular pulse.
    For generating time-based transitions in functions.

    Parameters:
        t (float or np.ndarray): The times at which to evaluate the function (24 hours in this case)
        a (float): Start time of the box
        b (float): End time of the box
        k (float): Sharpness of the sigmoid edges. Higher k means steeper transitions

    Returns:
        float or np.ndarray: Value(s) of the box function at time t -- ranging from 0 to about 1.
    """
    return 1 / (1 + np.exp(-k * (t - a))) - 1 / (1 + np.exp(-k * (t - b)))

def bile_salt_function(t, k=10):
    '''
    Simulates bile salt concentration across a 24-hour circadian cycle.
    Function adds three postprandial peaks (around 9, 13, and 19) using smoothed box
    function from above to model bile salt increases due to meal times (8, 12, 18). This represents an idealized
    bile analysis with sharp rises and gradual falls in concentration.

    Parameters:
        t (float or np.ndarray): Time(s) in hours (typically 0 to 24).
        k (float): Sharpness of the box function edges (sigmoid steepness).

    Returns:
        float or np.ndarray: Simulated bile salt concentration at time t.
    '''
    base = 4 #Baseline bile salt rate: 4 micromoles/L
    peaks = [9, 13, 19] #Maximums here at 16 mM/L, as peaks occur about 60 min after mealtimes (8, 12, 18)
    value = base
    for h in peaks:
        value += 12 * box(t, h, h+1, k) # Go down to 12 mM the first hour after peak (gradual decrease)
        value += 8 * box(t, h+1, h+2, k) # Go down to 8 mM the second hour after peak (gradual decrease)
    return value


def param_sin_func(t, parameters):
    """
    A parameterized single sine function. Represents one part of a 8-sine approximation of the bile derivativee
    Parameters:
        t (float or np.ndarray): Time in hours
        parameters (list of 3 floats): (phase_shift, frequency, amplitude)

    Returns:
        float/np.ndarray: Value of the sine function at time t
    """
    freq_shift, freq, scale = parameters
    return scale * np.sin(freq * t + freq_shift)

def metric_of_success(parameters):
    """
    function for optimization: mean squared error between target + predicted derivatives
    Measures how well the set of 8 sine functions (with 24 parameters) fits true derivative of the bile concentration over time.

    Parameters:
        parameters (list of 24 floats): Parameters for 8 sine functions (3 per sine)

    Returns:
        float: Mean squared error between modeled derivative and observed.
    """

    return np.mean((param_sin_func(time, parameters) - signal)**2)

def sum8_sin_func(t, parameters):
    """
    Sum of 8 sine functions for approximating the derivative of bile concentration. Reconstructs a smooth, periodic signal by summing eight separate
    sine components, each with different amplitude, frequency, and phase.

    Parameters:
        t (float or np.ndarray): Time(s) in hours
        parameters (list of 24 floats): Each sine has 3 parameters (8 x 3 = 24 total),
            ordered as [shift1, freq1, amp1, ..., shift8, freq8, amp8]

    Returns:
        float or np.ndarray: Value of the summed sine approximation at time t
    """
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
        entry is a dummy variable that keeps track of time. So at t=0, the second entry should be 0. (list or np.ndarray of 2 floats)
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

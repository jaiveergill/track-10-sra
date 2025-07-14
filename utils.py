import numpy as np
from generate_bile_salt_params import bile_salt_function, sum8_sin_func, param_sin_func

best_parameters = np.load("params/best_parameters.npy")
t_values = np.linspace(0, 24, 1000)
bile_values = [bile_salt_function(t) for t in t_values]

# Compare to the original bile concentration profile
def bile_salt_function_differential_equation(x, t):
    """ Differential equation describing the nonlinear oscillation of bile concentration over a 24 hour cycle. Redefined from bile_salt.py for use in the ODE solver.
    @param x: a 1D array of length 2 of floats, the first entry is the molar bile concentration and the second
        entry is a dummy variable that keeps track of time. So at t=0, the second entry should be 0.
    @param t: nonegative float, the time at which to evaluate the differential equation.
    @returns a 1D array of length 2 of floats, the time derivative of the bile concentration and the dummy variable.
    """
    bile, y = x #bile2 and bile1
    bile = sum8_sin_func(y, best_parameters) # Our model for the dynamics
    return np.array([bile, 1])
    

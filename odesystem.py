#hi 

"""
This script contains the functions for the differential equations used in the system.
"""

import numpy as np
from generate_bile_salt_params import bile_salt_function, sum8_sin_func, param_sin_func

# Load params generated in generate_bile_salt_param
best_parameters = np.load("params/best_parameters.npy")

# example time series data
t_values = np.linspace(0, 24, 1000)
bile_values = [bile_salt_function(t) for t in t_values]

def mu(t, mu_max, F, S, K_s, m,):
    """
    Equation:
        µ(t) = µ_max * F(t) * (S/K_s+S) - m
        
        µ_max * S/(K_s+S) - Monod term for growth
        m - maintainance/death rate

    """
    return mu_max * F * (S / (K_s + S))

def dN_dt(mu, N, m, epsilon, S_out):
    """
    Computes the absolute rate of change of the population at time t
    This is the absolute growth rate, which is the absolute rate of change of the population in cells

    Equation:
        dN/dt = (mu - m) * N - epsilon * S_out
            mu: specific growth rate (1/time)
            m: maintainance/death rate
            N: population
            episolon * S_out: amount of cells that are flushed out with substrate being flushed out
    """
    return (mu - m) * N - epsilon*S_out

def dS_dt(N, S, mu_max, K_s, Y, m, S_in):
    """
    Computes the rate of change of substrate concentration using Monod kinetics,
    maintenance, and an external feed term.

    Equation:
        dS/dt = - (μ * N) / Y  -  m * N  +  S_in
    """
    mu_ = mu_max * S / (K_s + S)
    return - (mu_ * N) / Y - m * N + S_in


def dF_dt(H_val, E, c_p, F, k_F):
    """
    Equation:
        - dF/dt = H * (1-E) * (1-c_p) - k_F * F
        HGT rate * stress * plasmid cost - decay value of fitness
    """
    growth = H_val * (1 - E) * (1 - c_p)
    decay  = -k_F * F
    return growth + decay

def dE_dt(theta_pH, Z_pH, theta_temp, Z_temp, theta_bile, Z_bile, E, k_E):
    """
    Equation:
        - dE/dt = (theta_pH*Z_pH + theta_temp*Z_temp + theta_bile*Z_bile) / (theta_pH + theta_temp + theta_bile) - decay
        decay  = -k_E * E
    """
    raw_stress = (theta_pH*Z_pH + theta_temp*Z_temp + theta_bile*Z_bile) / (theta_pH + theta_temp + theta_bile)
    decay   = -k_E * E
    return raw_stress + decay

def Z_pH(pH, pH_opt, sigma_pH):
    """
    Computes the pH environemental stress factor using Gaussian based penalty.
    
    Equation:
        Z_pH = exp[-((pH - pH_opt)^2) / (2 * sigma_pH^2)]
    
    
    @Params
        pH: current pH value
        pH_opt: optimal pH value (prob around 6.7-7)
        sigma_pH: standard deviation for the penalty for Guassian curve
        
    @returns
        pH-based environemental stress
    """
    return np.exp(-((pH - pH_opt) ** 2) / (2 * sigma_pH ** 2))

def Z_temp(T, T_opt, sigma_T):
    """
    Computes the temperature based environemental stress factor using Gaussian based penalty.
    
    Equation:
        Z_temp = exp[-((T - T_opt)^2) / (2 * sigma_T^2)]
    
    @Params
        T: current temperature
        T_opt: optimal temperature value (37 deg C)
        sigma_T: standard deviation for the penalty for Guassian curve
        
    @returns
        pH-based environemental stress
    """
    return np.exp(-((T - T_opt) ** 2) / (2 * sigma_T ** 2))

def bile_salt_function_differential_equation(x, t):
    """
    Differential equation describing the nonlinear oscillation of bile concentration over a 24 hour cycle.
    Redefined from bile_salt_function for use in an ODE solver.
    
    @param x: a 1D array of length 2 of floats, where the first entry is the bile concentration and 
              the second is a dummy variable tracking time (initially 0 at t=0).
    @param t: nonnegative float, the time at which the derivative is evaluated.
    @returns: a 1D array of length 2 of floats representing the derivative of bile concentration and the dummy variable.
    """
    bile, y = x  # bile concentration and dummy time variable
    bile = sum8_sin_func(y, best_parameters)  # Compute bile based on sinusoidal model
    return np.array([bile, 1])

def Z_bile(bile, bile_opt, sigma_bile):
    """
    Computes the bile based environemental stress factor using Gaussian based penalty.
    
    Equation:
        Z_bile = exp[-((bile - bile_opt)^2) / (2 * sigma_bile^2)]
    
    @Params
        bile: current bile salt concentration
        bile_opt: optimal bile salt concentration
        sigma_bile: standard deviation for the penalty for Guassian curve
        
    @returns
        Bile-based environemental stress
    """
    return np.exp(-((bile - bile_opt) ** 2) / (2 * sigma_bile ** 2))

def H(t, beta_max, D, R, S, K_s, E, K_c):
    """
    Equation:
        H(t) = ß_max * (D*R/(K_c + D*R))
        * ((S/(K_s+S))) (monod)
        * (1-E) (stress term)

    """
    return beta_max * D * R / (K_c + D * R) * S / (K_s + S) * (1 - E)


def dD_dt(mu, D, H_val, c_p, lambda_):

    return mu * (1 - c_p) * D + H_val - lambda_ * D


def dR_dt(mu, R, H_val, lambda_, D):

    return mu * R - H_val + lambda_ * D


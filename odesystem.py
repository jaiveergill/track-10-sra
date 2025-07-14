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

def mu(t, mu_max, F, N, K, epsilon, dS_dt):
    """
    Computes the current growth rate of bacteria at a given time.
    This is the specific growth rate which shows how the population grows per unit of population, so it is different from dN_dt
    
    Equation:
        mu(t) = mu_max * F(t) * (1 - N(t)/K) - epsilon * (dS/dt)

    @Params
        t: current time
        mu_max: maximum growth rate
        F: fitness adjustment factor
        N: total population
        K: carrying capacity
        epsilon: rate of bacteria leaving with the substrate
        dS_dt: derivative of substrate concentration

    @returns:
        Growth rate
    """
    return mu_max * F * (1 - N / K) - epsilon * dS_dt

def dN_dt(mu, N):
    """
    Computes the absolute rate of change of the population at time t
    This is the absolute growth rate, which is the absolute rate of change of the population in cells

    Equation:
        dN/dt = mu * N(t)
    """
    return mu * N

def dS_dt(dN_dt, Y, S_in):
    """
    Computes the derivative of substrate concentration.
    
    Equation:
        dS/dt = - (1/Y) * dN/dt + S_in(t)
    
    @Params
        dN_dt: derivative of the population
        Y: biomass yield coefficient (number of cells produced by one gram of substrate so that 1/Y = substrate needed for one cell)
        S_in: rate of substrate being added back into the system
        TODO: Make S_in an oscillating function
        
    @returns
        Substrate concentration rate of change
    """
    return - (1 / Y) * dN_dt + S_in

def dF_dt(H_val, E, c_p):
    """
    Computes the rate of change of the fitness adjustment factor.
    
    Equation:
        dF/dt = H(t) * (1 - E(t)) * (1 - c_p)
   
    @Params
        H_val: rate of horizontal gene transfer
        E: environmental stress factor
        c_p: the metabolic cost of the plasmid
        
    @returns
        Fitness adjustment rate (float)
    """
    return H_val * (1 - E) * (1 - c_p)

def dE_dt(theta_pH, Z_pH_val, theta_temp, Z_temp_val, theta_bile, Z_bile_val):
    """
    Computes the rate of change of the environmental stress factor.
    
    Equation:
        dE/dt = theta_pH * Z_pH + theta_temp * Z_temp + theta_bile * Z_bile, constrained by [0, 1]
    
    @Params
        theta_pH: weight for pH stress
        theta_temp: weight for temperature stress
        theta_bile: weight for bile stress
        Z_pH_val: effect of pH on environemental stress
        Z_temp_val: effect of temp on environemental stress
        Z_bile_val: effect of bile on environemental stress
        
    @returns
        Environmental stress rate
    """
    return (theta_pH * Z_pH_val + theta_temp * Z_temp_val + theta_bile * Z_bile_val) / 3 # I added divide by 3 to limit the term to [0, 1], might have to change this in paper as well

def H(t, beta_max, D, R, S, K_s, E):
    """
    Computes the rate of HGT
    
    Equation:
        H(t) = (beta_max * D(t) * R(t) * S(t)) / (K_s + S(t)) * (1 - E(t))
    
    @Params
        t: current time
        beta_max: maximum rate of HGT
        D: amount of donor cells
        R: amount of recipient cells
        S: substrate concentration
        K_s: half-saturation constant
        E: environmental stress factor (computed from E(t))
        
    @returns
        Horizontal gene transfer rate (float)
    """
    return (beta_max * D * R * S) / (K_s + S) * (1 - E)

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

def dD_dt(dN_dt, D, N, H_val, c_p):
    """
    Computes the derivative of the donor cell population.
    
    Equation:
        dD/dt = (1 - c_p) * dN/dt * (D/N) + H(t)
   
    @Params
        dN_dt: derivative of the total population
        D: donor cell population at the time t
        N: total cell population at the time t
        H_val: rate of HGT
        c_p: the metabolic cost of the plasmid
        
    @returns
        Donor cell population derivative (float)
    """
    return (1 - c_p) * dN_dt * (D / N) + H_val

def dR_dt(dN_dt, R, N, H_val):
    """
    Computes the derivative of the recipient cell population.
    
    Equation:
        dR/dt = dN/dt * (R/N) - H(t)
    
    @Params
        dN_dt: derivative of the total population
        R: recipient cell population at the time t
        N: total cell population at the time t
        H_val: rate of HGT
        
    @returns
        Recipient cell population derivative (float)
    """
    return dN_dt * (R / N) - H_val

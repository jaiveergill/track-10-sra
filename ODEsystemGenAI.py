import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ─── Bile salt concentration model (time‐invariant via dummy state z) ───
def logistic(x, k=10):
    return 1 / (1 + np.exp(-k * x))

def box_function(z, a, b, k=10):
    return logistic(z - a, k) - logistic(z - b, k)

def bile_salt_concentration(z):
    base = 4.0
    pulses = [(9, 10), (13, 14), (19, 20)]
    val = base
    for start, end in pulses:
        val += 12 * box_function(z, start, start+1)
        val += 8  * box_function(z, start+1, end)
    return val

def bile_salt_derivative(z):
    k = 10
    deriv = 0.0
    pulses = [(9, 10), (13, 14), (19, 20)]
    for start, end in pulses:
        sig1 = logistic(z - start, k)
        sig2 = logistic(z - (start+1), k)
        d_box1 = k * (sig1 * (1 - sig1) - sig2 * (1 - sig2))
        sig3 = logistic(z - (start+1), k)
        sig4 = logistic(z - end, k)
        d_box2 = k * (sig3 * (1 - sig3) - sig4 * (1 - sig4))
        deriv += 12 * d_box1 + 8 * d_box2
    return deriv

# ─── Parameters ───
mu_max  = 1.0
K       = 1.0
epsilon = 0.05
Y       = 0.5
theta1  = -1.0 / Y
theta2  = 0.1
theta3  = -0.05
theta4  = 0.325
theta5  = 0.1
c       = 0.1

# ─── Autonomous ODE system ───
def ode_system(z, y):
    N, S, E, F, H, dummy = y
    B   = bile_salt_concentration(dummy)
    dB  = bile_salt_derivative(dummy)
    dSdz = theta1 * N + theta2 * dB
    dNdz = mu_max * N * F * (1 - N/K) - epsilon * dSdz
    dEdz = theta3 * B + theta4
    dFdz = H * (1 - E) * (1 - c)
    dHdz = theta5 * N * dSdz
    d_dummy = 1.0
    return [dNdz, dSdz, dEdz, dFdz, dHdz, d_dummy]

# ─── Initial conditions and integration ───
y0 = [0.01, 2.1, 0.5, 0.9, 1.0, 0.0]  # [N, S, E, F, H, dummy]
z_span = (0, 24)
z_eval = np.linspace(0, 24, 1000)

sol = solve_ivp(ode_system, z_span, y0, t_eval=z_eval, method="RK45",
                rtol=1e-6, atol=1e-9)

# Extract states
N_vals = sol.y[0]
dS_vals = theta1 * N_vals + theta2 * bile_salt_derivative(sol.y[5])
growth_rate = mu_max * N_vals * sol.y[3] * (1 - N_vals/K) - epsilon * dS_vals

# ─── Plot: Growth rate ───
plt.figure(figsize=(8, 4))
plt.plot(sol.t, growth_rate)
plt.xlabel("Time (hours)")
plt.ylabel("Growth rate dN/dt")
plt.title("Autonomous Bacterial Growth Rate Over 24 h")
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── Plot: Population size (fraction of K) ───
plt.figure(figsize=(8, 4))
plt.plot(sol.t, N_vals / K)
plt.xlabel("Time (hours)")
plt.ylabel("Population size (fraction of carrying capacity)")
plt.title("Bacterial Population Size Over 24 h")
plt.grid(True)
plt.tight_layout()
plt.show()

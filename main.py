import numpy as np
from scipy.integrate import odeint
import odesystem
import matplotlib.pyplot as plt
from constants import BILE_OPT, SIGMA_BILE

t_values = np.linspace(0, 24, 1000)
best_parameters = np.load("params/best_parameters.npy")

simulated_bile = odeint(bile_salt_function_differential_equation, [4, 0], t_values, args=(best_parameters,))
plt.plot(t_values, simulated_bile[:, 0], label="Simulated Bile Concentration")
plt.plot(t_values, Z_bile(simulated_bile[:, 0], BILE_OPT, SIGMA_BILE), label="Z_bile Bile Concentration")
plt.xlabel("Time (h)")
plt.ylabel("Bile concentration")
plt.title("Time Series Signal with Simulated Bile Concentration")
plt.legend()
plt.show()

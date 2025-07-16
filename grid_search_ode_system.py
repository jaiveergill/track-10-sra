import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# USER-SUPPLIED bile waveform pieces -------------------------
# ------------------------------------------------------------
try:
    from generate_bile_salt_params import param_sin_func
    best_parameters = np.load("params/best_parameters.npy")
except Exception as e:
    print("WARN: Could not import param_sin_func/best_parameters; using fallback simple sine.")
    def param_sin_func(t, p):
        # p: [amp, freq, phase]; fallback harmless
        amp, freq, phase = p
        return amp * np.sin(2*np.pi*freq*t + phase)
    best_parameters = np.array([0.5,1/24,0, 0.2,2/24,0, 0.1,3/24,0])

def sum8_sin_func(t, parameters=best_parameters):
    """User's bile forcing fit; returns dimensionless bile 'signal'."""
    # Assume parameters length multiple of 3
    nterm = len(parameters)//3
    out = 0.0
    for i in range(nterm):
        out += param_sin_func(t, parameters[3*i:3*(i+1)])
    return out

def bile_waveform(t):
    """Periodic bile forcing over 24h."""
    return sum8_sin_func(t % 24, best_parameters)

# ------------------------------------------------------------
# BASE PARAMETERS (global defaults; can override per sweep) ---
# ------------------------------------------------------------
mu_max  = 1.0      # 1/h
Y       = 0.5      # biomass/substrate
epsilon = 0.05     # user-chosen coupling to bile rate in growth term
theta1  = -1.0 / Y # substrate consumption scaling
theta2  = 0.14     # bile influence on substrate
theta5_default = 1e-4  # host coupling

# Initial conditions (reduced state: N, S, H, tclock)
N0 = 0.5
S0 = 1.1
H0 = 0.5
y0_reduced = [N0, S0, H0, 0.0]

# ------------------------------------------------------------
# ENVIRONMENT → theta4 mapping  ------------------------------
# (your Gaussian "baseline_drift" used as a simple stress proxy)
# ------------------------------------------------------------
PH_OPT        = 6.8
TEMP_OPT      = 37.0
SIGMA_PH_DR   = 0.5    # pH units
SIGMA_TEMP_DR = 2.0    # °C
THETA4_MAX    = 0.325

def baseline_drift(pH, temp,
                   theta4_max=THETA4_MAX,
                   pH_opt=PH_OPT, sigma_pH=SIGMA_PH_DR,
                   temp_opt=TEMP_OPT, sigma_temp=SIGMA_TEMP_DR):
    ph_term   = np.exp(-((pH   - pH_opt)**2)    / (2*sigma_pH**2))
    temp_term = np.exp(-((temp - temp_opt)**2)  / (2*sigma_temp**2))
    return theta4_max * ph_term * temp_term

# ------------------------------------------------------------
# INSTANTANEOUS STRESS & FITNESS -----------------------------
# ------------------------------------------------------------
def stress_from_env(B, theta3, theta4):
    """
    Linear bile contribution around a baseline env stress.
    Bound result to [0,1] so downstream terms stay interpretable.
    """
    return np.clip(theta4 + theta3 * B, 0.0, 1.0)

def fitness_from_stress(E, H, c):
    """
    Effective fitness multiplier (dimensionless) entering growth term.
    c >0 => cost; c <0 => benefit.
    """
    return (1.0 - c) * (1.0 - E) * H

# ------------------------------------------------------------
# ODE (reduced: we do NOT integrate E or F; computed on the fly)
# ------------------------------------------------------------
def ode_system(t, y, mu_max, theta1, theta2, theta3, theta4, c, theta5):
    N, S, H, tclock = y

    # bile forcing @ current internal clock
    B = bile_waveform(tclock)
    # user wanted dN_dt ~ -epsilon * dB_dt ; we approximate with instantaneous B signal.
    dB_dt = B

    # instantaneous stress & fitness
    E = stress_from_env(B, theta3, theta4)
    F = fitness_from_stress(E, H, c)

    # substrate & growth (retain your structure)
    dS_dt = theta1 * N * S/(1.0 + S) * mu_max + theta2 * dB_dt
    dN_dt = mu_max * N * F * (S/(1.0 + S)) - epsilon * dB_dt
    dH_dt = theta5 * N * dS_dt * E  # leave as-is; tiny by default
    dtclock_dt = 1.0                # advance internal clock (h)

    return [dN_dt, dS_dt, dH_dt, dtclock_dt]

# ------------------------------------------------------------
# SIMULATION WRAPPER -----------------------------------------
# ------------------------------------------------------------
def simulate_once(Tend,
                  mu_max=mu_max,
                  theta1=theta1,
                  theta2=theta2,
                  theta3=-0.05,
                  theta4=0.325,
                  c=0.1,
                  theta5=theta5_default,
                  y0=y0_reduced,
                  t_eval=None,
                  rtol=1e-6,
                  atol=1e-9):
    """
    Integrate model from t=0..Tend (h) and return (t, sol, derived metrics).
    """
    if t_eval is None:
        t_eval = np.linspace(0, Tend, 500)

    args = (mu_max, theta1, theta2, theta3, theta4, c, theta5)
    sol = solve_ivp(lambda t,y: ode_system(t,y,*args),
                    (0, Tend), y0, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    N = sol.y[0]
    S = sol.y[1]
    H = sol.y[2]
    tclock = sol.y[3]

    # reconstruct bile, stress, fitness time courses
    B = bile_waveform(tclock)
    E = stress_from_env(B, theta3, theta4)
    F = fitness_from_stress(E, H, c)

    # instantaneous growth rate (same structure used in ODE)
    dB = B  # consistent with ODE
    dS = theta1 * N * S/(1.0 + S) * mu_max + theta2 * dB
    growth_rate = mu_max * N * F * (S/(1.0 + S)) - epsilon * dB

    # summary metrics
    N_final = N[-1]
    N_max   = N.max()
    gr_mean = np.trapz(growth_rate, t) / (t[-1] - t[0])
    F_mean  = F.mean()
    E_mean  = E.mean()

    metrics = dict(N_final=N_final,
                   N_max=N_max,
                   mean_growth=gr_mean,
                   mean_F=F_mean,
                   mean_E=E_mean)
    traces = dict(t=t, N=N, S=S, H=H, B=B, E=E, F=F, growth_rate=growth_rate)
    return metrics, traces

# ------------------------------------------------------------
# CONFIG: parameter sweeps -----------------------------------
# Adjust ranges here. These are biologically informed but broad.
# ------------------------------------------------------------
TEND_H = 24.0

# plasmid cost: beneficial -> strongly deleterious
c_grid = np.linspace(-0.2, 0.3, 26)   # step 0.02

# bile->stress slope; negative = bile protective, positive = bile stressful
theta3_grid = np.linspace(-0.1, 0.1, 41)  # fine grid

# environmental pH / temperature grid (maps to theta4 via baseline_drift)
pH_grid   = np.linspace(5.5, 8.0, 11)
Temp_grid = np.linspace(32.0, 40.0, 9)

# choose which summary metric to visualize in heatmaps
METRIC_NAME = "N_final"   # options: N_final, N_max, mean_growth, mean_F, mean_E

# ------------------------------------------------------------
# GRID SWEEP ENGINE ------------------------------------------
# We'll produce 3 sets of heatmaps:
#   1) c vs theta3 at 3 representative environments (acid/cool, neutral/opt, alkaline/warm)
#   2) pH vs Temp at 3 representative plasmid costs (beneficial, neutral, costly)
# Feel free to comment out blocks you don't need.
# ------------------------------------------------------------

def sweep_c_theta3_over_env(env_list=None):
    if env_list is None:
        env_list = [
            ("AcidCool",   5.8, 32.0),
            ("NearOpt",    6.8, 37.0),
            ("AlkWarm",    7.8, 40.0),
        ]
    n_env = len(env_list)
    fig, axes = plt.subplots(1, n_env, figsize=(4*n_env, 4), sharey=True)
    if n_env == 1:
        axes = [axes]

    for ax, (ename, pH, temp) in zip(axes, env_list):
        theta4 = baseline_drift(pH, temp)
        Z = np.zeros((len(c_grid), len(theta3_grid)))
        for i, cval in enumerate(c_grid):
            for j, th3 in enumerate(theta3_grid):
                m, _ = simulate_once(TEND_H, theta3=th3, theta4=theta4, c=cval)
                Z[i,j] = m[METRIC_NAME]

        im = ax.imshow(Z, origin="lower", aspect="auto",
                       extent=[theta3_grid[0], theta3_grid[-1],
                               c_grid[0], c_grid[-1]],
                       cmap="rainbow")
        ax.set_title(f"{ename}\npH={pH:.1f},T={temp:.1f}°C")
        ax.set_xlabel("theta3 (bile→stress slope)")
        ax.set_ylabel("c (plasmid cost)")
        fig.colorbar(im, ax=ax, shrink=0.8, label=METRIC_NAME)

    fig.suptitle(f"{METRIC_NAME} after {TEND_H:.0f} h: c vs theta3 across environments")
    fig.tight_layout()
    return fig

def sweep_env_over_c(c_list=None):
    if c_list is None:
        c_list = [
            ("Beneficial", -0.1),
            ("Neutral",     0.0),
            ("Costly",      0.2),
        ]
    n_c = len(c_list)
    fig, axes = plt.subplots(1, n_c, figsize=(4*n_c,4), sharey=True)
    if n_c == 1:
        axes = [axes]

    for ax, (cname, cval) in zip(axes, c_list):
        Z = np.zeros((len(pH_grid), len(Temp_grid)))
        for i, pH in enumerate(pH_grid):
            for j, temp in enumerate(Temp_grid):
                theta4 = baseline_drift(pH, temp)
                m, _ = simulate_once(TEND_H, theta3=-0.05, theta4=theta4, c=cval)  # default theta3; change if needed
                Z[i,j] = m[METRIC_NAME]

        im = ax.imshow(Z, origin="lower", aspect="auto",
                       extent=[Temp_grid[0], Temp_grid[-1],
                               pH_grid[0], pH_grid[-1]],
                       cmap="rainbow")
        ax.set_title(f"{cname} plasmid (c={cval:+.2f})")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("pH")
        fig.colorbar(im, ax=ax, shrink=0.8, label=METRIC_NAME)

    fig.suptitle(f"{METRIC_NAME} after {TEND_H:.0f} h: Environment sweep")
    fig.tight_layout()
    return fig

# ------------------------------------------------------------
# RUN DEFAULT SWEEPS WHEN FILE EXECUTED ----------------------
# ------------------------------------------------------------
if __name__ == "__main__":
    fig1 = sweep_c_theta3_over_env()
    fig2 = sweep_env_over_c()
    plt.show()
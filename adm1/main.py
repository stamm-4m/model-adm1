"""
ADM1 Reactor Simulation (Adapted from PyADM1)

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from src.reactor import ADM1Reactor
from src.parameters import ADM1Parameters
from src.influent import Influent 
import time

# -----------------------------
# 1. Load parameters and influent
# -----------------------------
param = ADM1Parameters("configs/adm1_parameters.yaml")
constants = None
reactor = ADM1Reactor(param, constants)

influent = Influent("configs/daily_averages.csv")

# Set initial state vector (all ones as placeholder)
y0 = np.ones(38)

# -----------------------------
# 2. Define time span for simulation
# -----------------------------

time_data = influent.get_time()

t_span = (0, len(time_data)-1)
t_eval = np.arange(*t_span)

# -----------------------------
# 3. Wrapper to update influent state at each step
# -----------------------------

def ADM1_wrapper(t, y):
    step = int(t)
    step = min(step, len(influent.get_time())-1)
    print(f"t = {influent.get_time()[step]}")
    reactor.influent_state = influent.get(step)

    return reactor.ADM1_ODE(t, y)

# -----------------------------
# 4. Solve ODE
# -----------------------------
start_time = time.time()
sol = solve_ivp(
    fun=ADM1_wrapper,
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method="RK45" #, # 'DOP853', "BDF"
    #rtol=1e-6,
    #atol=1e-8
)
end_time = time.time()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"\nSimulation time: {minutes} minutes {seconds} seconds")

# -----------------------------
# 5. Plot some key outputs
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[6], label="Acetate (S_ac)")
plt.plot(sol.t, sol.y[8], label="Methane (S_ch4)")
plt.plot(sol.t, sol.y[7], label="Hydrogen (S_h2)")
plt.xlabel("Time [days]")
plt.ylabel("Concentration [g/L]")
plt.title("ADM1 Reactor Simulation")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 5. Svame simulations in csv file
# -----------------------------

# Transpose state matrix
states = sol.y.T

# Column names (must match your ADM1 ordering)
state_names = [
    "S_su","S_aa","S_fa","S_va","S_bu","S_pro","S_ac",
    "S_h2","S_ch4","S_IC","S_IN","S_I",
    "X_xc","X_ch","X_pr","X_li","X_su","X_aa","X_fa",
    "X_c4","X_pro","X_ac","X_h2","X_I",
    "S_cation","S_anion",
    "S_H_ion","S_va_ion","S_bu_ion","S_pro_ion","S_ac_ion",
    "S_hco3_ion","S_nh3",
    "S_gas_h2","S_gas_ch4","S_gas_co2",
    "S_co2","S_nh4_ion"
]

# Create DataFrame
df = pd.DataFrame(states, columns=state_names)

# Add time column
df.insert(0, "time", sol.t)

# Save to CSV
df.to_csv("results/dynamic_out.csv", index=False)

print("dynamic_out.csv written successfully ✅")
"""
Example — Tier 1: replace the acetoclastic methanogenesis rate Rho_11.

This file is intentionally minimal. It shows the *plug-in pattern* — the
function is read from the YAML config at startup time and called in place
of the classical Monod expression at every solver step. In a real hybrid
project, replace the body with a call to a sklearn / PyTorch / ONNX model.

Wired by:
  configs/Scenario.yaml > scenarios.<your_scenario>.hybrid.rate_overrides:
      Rho_11: "examples.hybrid_rate_example:acetoclastic_rate_T_aware"

Required signature
------------------
def f(state: dict, inhib: dict, param) -> float

  state : dict of all 38 ADM1 state variables (S_su, S_ac, X_ac, ...)
          plus the acid-base species (S_H_ion, S_nh3, S_co2, ...).
  inhib : dict of inhibition factors {I_5..I_12, I_nh3} after any other
          inhibition_overrides have already been applied.
  param : ADM1Parameters object — read attributes like param.k_m_ac,
          param.K_S_ac, param.T_op.

Returns
-------
float : Rho_11 in kgCOD.m^-3.d^-1.
"""

import numpy as np


def acetoclastic_rate_T_aware(state: dict, inhib: dict, param) -> float:
    """
    Temperature-aware acetoclastic methanogenesis rate.

    Reproduces the classical Monod form, then multiplies by a Gaussian
    temperature factor centred at T_opt = 308.15 K (mesophilic) — meant
    to illustrate that you can blend physical structure with a learned
    correction. Replace the temperature_factor line with your model
    output.
    """
    S_ac = state["S_ac"]
    X_ac = state["X_ac"]
    I_11 = inhib["I_11"]

    classical_monod = (
        param.k_m_ac
        * S_ac / (param.K_S_ac + S_ac)
        * X_ac
        * I_11
    )

    T_op = param.T_op
    T_opt = 308.15            # K, mesophilic optimum
    bandwidth = 15.0          # K, falloff scale
    temperature_factor = float(np.exp(-((T_op - T_opt) / bandwidth) ** 2))

    return classical_monod * temperature_factor

"""
Example — Tier 1: replace the free-ammonia inhibition factor I_nh3.

The classical ADM1 form is the simple non-competitive
    I_nh3 = 1 / (1 + S_nh3 / K_I)
This example swaps it for a Hill-style sigmoid with a tunable exponent n,
which gives a sharper transition around K_I.

Wired by:
  configs/Scenario.yaml > scenarios.<your_scenario>.hybrid.inhibition_overrides:
      I_nh3: "examples.hybrid_inhibition_example:hill_nh3_inhibition"

Required signature
------------------
def f(state: dict, param) -> float

  state : dict of all 38 ADM1 state variables (read S_nh3 here).
  param : ADM1Parameters object — read attributes like param.K_I_nh3.

Returns
-------
float in [0, 1]: 1.0 means no inhibition, 0.0 means full inhibition.
"""


def hill_nh3_inhibition(state: dict, param) -> float:
    """
    Hill-style ammonia inhibition.
        I_nh3 = 1 / (1 + (S_nh3 / K_I) ** n)
    With n = 1 this collapses to the classical ADM1 form. Larger n gives
    a steeper transition. In a learned model, n (or the entire mapping)
    would be fit from observations.
    """
    n = 2.0
    K_I = param.K_I_nh3
    S_nh3 = max(state["S_nh3"], 0.0)
    ratio = S_nh3 / K_I
    return 1.0 / (1.0 + ratio ** n)

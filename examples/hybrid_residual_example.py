"""
Example — Tier 2: a UDE-style residual correction added to dy/dt.

Universal Differential Equation pattern:
    dy/dt = ADM1(y, t) + f_theta(y, t)

The classical ADM1 derivatives are computed first; the function below is
called once per solver step and its output is *added* to the classical
dy/dt vector. Use it to encode a small learned correction that captures
unmodelled effects (e.g. a feedstock-specific bias, a temperature drift,
or a sensor-bias compensation).

Wired by:
  configs/Scenario.yaml > scenarios.<your_scenario>.hybrid.residual_correction:
      "examples.hybrid_residual_example:methane_bias_correction"

Required signature
------------------
def f(t: float, state: dict, param) -> dict | numpy.ndarray

  t     : current simulation time (d).
  state : dict of all 38 ADM1 state variables (S_su, S_ac, S_ch4, ...)
          plus the acid-base species (S_H_ion, S_nh3, ...).
  param : ADM1Parameters object.

Two return formats are accepted:
  1) dict {state_name: dydt_delta}  — only listed states get a correction
                                       (most ergonomic, recommended)
  2) numpy.ndarray of length 38     — aligned with FULL_STATE_NAMES
                                       (use for batch/vectorised models)
"""


def methane_bias_correction(t: float, state: dict, param) -> dict:
    """
    Add a small constant production bias to dissolved methane.

    Mimics a learned correction term that captures unmodelled
    extra CH4 generation. In practice you would replace 0.001 with the
    output of a model trained against plant data.
    """
    return {
        "S_ch4": 0.001,   # +0.001 kgCOD.m^-3.d^-1 on dissolved methane
    }


def zero_residual(t: float, state: dict, param) -> dict:
    """
    No-op residual — useful to test the wiring without changing dynamics.
    Returns an empty dict, so the classical ADM1 trajectory is unchanged.
    """
    return {}

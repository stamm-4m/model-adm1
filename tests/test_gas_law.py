"""
Tests for the gas-flow law switch (configs/adm1_parameters.yaml -> gas_flow.gas_law_patm).

Run from the repo root:   python tests/test_gas_law.py

1. gas_law_patm = 0 (default) must reproduce the validated BSM2 right-hand side exactly, including the
   gas phase (reference values frozen below from commit 74fa6bf, BSM2 initial state, BSM2 constant feed).
2. The vectorised post-processing (compute_gas_flow_rate / compute_gas_outputs) must return the same
   q_gas as the ODE, for both laws.
3. gas_law_patm = 1 must use q_gas = R T V_liq (rT8/16 + rT9/64 + rT10) / (P_atm - p_H2O)
   (Batstone et al. 2002; RM_without_Xc), written out independently here.
4. gas_law_patm = 1: at P_gas = P_atm the total headspace pressure must be stationary (dP_gas/dt = 0),
   and after a few days of simulation P_gas must have converged to P_atm.
5. Output conversions: q_gas_atm = q_gas P_gas/P_atm, q_gas_norm_dry and q_ch4_norm_dry consistent.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from src.reactor import FULL_STATE_NAMES, DYNAMIC_STATE_NAMES     # noqa: E402
from test_xc_flag import bsm2_state, make_reactor, equilibrated   # noqa: E402

GAS = ["S_gas_h2", "S_gas_ch4", "S_gas_co2"]


def gas_pressure_derivative(r, d):
    """dP_gas/dt [bar/d] from the derivatives of the three gas states."""
    p = r.param
    return p.R * p.T_op * (d["S_gas_h2"] / 16 + d["S_gas_ch4"] / 64 + d["S_gas_co2"])


def main():
    ok = True
    y_bsm2 = bsm2_state()

    # ---------------------------------------------------------------- 1. default law == validated
    r, p = make_reactor(1)
    y, _ = equilibrated(y_bsm2, p)
    d = dict(zip(FULL_STATE_NAMES, np.asarray(r.ADM1_ODE(0.0, y))))
    ref = {"S_gas_h2": 2.69404275099007e-06, "S_gas_ch4": 0.0001577408707174044,
           "S_gas_co2": 0.0008881218660570372, "S_h2": -0.0006190599006377407,
           "S_ch4": 0.0006250920943091653, "S_IC": 0.001595585870962625}
    worst = max(abs(d[n] - v) / abs(v) for n, v in ref.items())
    q_ref = 2577.0721035166866
    worst = max(worst, abs(r._last_gas["q_gas"] - q_ref) / q_ref)
    good = worst < 1e-12 and not r.gas_law_patm
    print(f"[1] gas_law_patm=0 (default) vs validated RHS + q_gas: max rel diff {worst:.1e} ->", "OK" if good else "FAIL")
    ok &= good

    # ---------------------------------------------------------------- 2. post-processing == ODE (both laws)
    for law in (0, 1):
        r, p = make_reactor(1, gas_law_patm=float(law))
        y, st = equilibrated(y_bsm2, p)
        r.ADM1_ODE(0.0, y)
        q_ode = r._last_gas["q_gas"]
        st = r._last_state            # the state the ODE actually used (acid-base re-solved inside the RHS)
        out = r.compute_gas_outputs(pd.DataFrame([st]))
        rel = abs(float(out["q_gas"][0]) - q_ode) / q_ode
        good = rel < 1e-13
        print(f"[2] gas_law_patm={law}: q_gas post-processing {float(out['q_gas'][0]):.6f} vs ODE {q_ode:.6f} "
              f"(rel {rel:.0e}) ->", "OK" if good else "FAIL")
        ok &= good

    # ---------------------------------------------------------------- 3. P_atm law formula
    r, p = make_reactor(1, gas_law_patm=1.0)
    y, _ = equilibrated(y_bsm2, p)
    r.ADM1_ODE(0.0, y)
    st = r._last_state
    ph2 = st["S_gas_h2"] * p.R * p.T_op / 16
    pch4 = st["S_gas_ch4"] * p.R * p.T_op / 64
    pco2 = st["S_gas_co2"] * p.R * p.T_op
    rT8 = p.k_L_a * (st["S_h2"] - 16 * r.K_H_h2 * ph2)
    rT9 = p.k_L_a * (st["S_ch4"] - 64 * r.K_H_ch4 * pch4)
    rT10 = p.k_L_a * (st["S_co2"] - r.K_H_co2 * pco2)
    q_indep = p.R * p.T_op * p.V_liq * (rT8 / 16 + rT9 / 64 + rT10) / (p.p_atm - r.p_gas_h2o)
    rel = abs(r._last_gas["q_gas"] - q_indep) / q_indep
    good = rel < 1e-13
    print(f"[3] gas_law_patm=1: q_gas {r._last_gas['q_gas']:.6f} vs RM formula {q_indep:.6f} (rel {rel:.0e}) ->",
          "OK" if good else "FAIL")
    ok &= good

    # ---------------------------------------------------------------- 4a. stationarity of P_gas at P_atm
    # scale the three gas states so that P_gas = P_atm exactly, keep the composition
    y4 = y.copy()
    P_dry = ph2 + pch4 + pco2
    scale = (p.p_atm - r.p_gas_h2o) / P_dry
    for n in GAS:
        y4[FULL_STATE_NAMES.index(n)] *= scale
    d4 = dict(zip(FULL_STATE_NAMES, np.asarray(r.ADM1_ODE(0.0, y4))))
    dP = gas_pressure_derivative(r, d4)
    dP_scale = p.R * p.T_op * abs(d4["S_gas_co2"]) + 1e-30      # size of one term, for a relative check
    good = abs(dP) / dP_scale < 1e-9
    print(f"[4a] gas_law_patm=1, P_gas = P_atm: dP_gas/dt = {dP:+.2e} bar/d ->", "OK" if good else "FAIL")
    ok &= good

    # ---------------------------------------------------------------- 4b. convergence P_gas -> P_atm
    r, p = make_reactor(1, gas_law_patm=1.0)
    y, _ = equilibrated(y_bsm2, p)                                   # BSM2 state: P_gas starts at ~1.065 bar
    y_dyn = np.array([y[FULL_STATE_NAMES.index(n)] for n in DYNAMIC_STATE_NAMES])
    r.expand_dynamic_state(y)                                         # set the template
    sol = solve_ivp(r.ADM1_ODE, (0.0, 5.0), y_dyn, method="BDF", rtol=1e-8, atol=1e-10)
    full = r.expand_dynamic_state(sol.y[:, -1])
    st_end = dict(zip(FULL_STATE_NAMES, full))
    P_end = sum(v for v in r.gas_state_to_partial_pressures(
        st_end["S_gas_h2"], st_end["S_gas_ch4"], st_end["S_gas_co2"])) + r.p_gas_h2o
    good = sol.success and abs(P_end - p.p_atm) < 1e-6
    print(f"[4b] gas_law_patm=1, 5 d from the BSM2 state: P_gas = {P_end:.8f} bar (P_atm {p.p_atm}) ->",
          "OK" if good else "FAIL")
    ok &= good

    # ---------------------------------------------------------------- 5. output conversions
    r, p = make_reactor(1)
    y, st = equilibrated(y_bsm2, p)
    out = {k: float(np.asarray(v)[0]) for k, v in r.compute_gas_outputs(pd.DataFrame([st])).items()}
    pch4 = st["S_gas_ch4"] * p.R * p.T_op / 64
    exp = {"q_gas_atm": out["q_gas"] * out["P_gas"] / p.p_atm,
           "q_gas_norm_dry": out["q_gas"] * (out["P_gas"] - r.p_gas_h2o) / 1.01325 * 273.15 / p.T_op,
           "q_ch4_norm_dry": out["q_gas"] * pch4 / 1.01325 * 273.15 / p.T_op}
    worst = max(abs(out[k] - v) / v for k, v in exp.items())
    good = worst < 1e-14 and out["q_ch4_norm_dry"] < out["q_gas_norm_dry"] < out["q_gas"] < out["q_gas_atm"]
    print(f"[5] q_gas {out['q_gas']:.1f} | q_gas_atm {out['q_gas_atm']:.1f} | q_gas_norm_dry {out['q_gas_norm_dry']:.1f}"
          f" | q_ch4_norm_dry {out['q_ch4_norm_dry']:.1f} m3/d ->", "OK" if good else "FAIL")
    ok &= good

    print("\nVERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

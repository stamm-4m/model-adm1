"""
Tests for the disintegration switch (configs/adm1_parameters.yaml -> disintegration.use_xc).

Run from the repo root:   python tests/test_xc_flag.py

1. use_xc = 1 must reproduce the validated BSM2 right-hand side exactly (reference values frozen
   below from the validated commit 7f2de9d, BSM2 initial state, BSM2 constant feed).
2. use_xc = 0 must conserve COD, carbon and nitrogen in the reaction terms (q_ad = 0), on the
   BSM2 state and on a biomass-only state (decay is the only path touched by the switch).
3. use_xc = 0 with f_*_xb not summing to 1 must be rejected.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.parameters import ADM1Parameters                       # noqa: E402
from src.reactor import ADM1Reactor, FULL_STATE_NAMES           # noqa: E402
from src.acid_base import compute_acid_base_equilibrium         # noqa: E402
import yaml                                                     # noqa: E402

COD_STATES = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_I",
              "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I"]
BIOMASS = ["X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2"]


def bsm2_state():
    """BSM2 initial state (key 'BSM2' of configs/Initial_states.yaml), independent of the active scenario."""
    raw = yaml.safe_load(open("configs/Initial_states.yaml", encoding="utf-8"))["BSM2"]
    return np.array([float(raw[n]["value"]) for n in FULL_STATE_NAMES])


def bsm2_feed():
    """BSM2 constant feed (key 'constant' of configs/Influent.yaml)."""
    raw = yaml.safe_load(open("configs/Influent.yaml", encoding="utf-8"))["constant"]
    return {k + "_in": float(v["value"]) for k, v in raw.items() if isinstance(v, dict) and "value" in v}


def make_reactor(use_xc, **overrides):
    p = ADM1Parameters(scenarios_file="__no_scenario__")  # FileNotFoundError -> base parameters only
    p.params["use_xc"] = float(use_xc)
    p.params.update(overrides)
    r = ADM1Reactor(p)
    r.influent_state = bsm2_feed()
    return r, p


def equilibrated(y, p):
    st = dict(zip(FULL_STATE_NAMES, y))
    st.update(compute_acid_base_equilibrium(st, p))
    return np.array([st[n] for n in FULL_STATE_NAMES]), st


def elemental_residuals(r, p, y):
    """Reaction-only derivatives (q_ad = 0): return the COD, C and N imbalance [per m3 per d]."""
    p.params["q_ad"] = 0.0
    y, st = equilibrated(y, p)
    d = dict(zip(FULL_STATE_NAMES, np.asarray(r.ADM1_ODE(0.0, y))))
    gas = r._last_gas
    # COD: liquid COD only leaves through gas transfer of H2 and CH4
    cod = sum(d[n] for n in COD_STATES) + gas["Rho_T_8"] + gas["Rho_T_9"]
    # Carbon: sum C_i * dX_i + dS_IC + CO2 and CH4 stripped to the gas phase
    C = {"S_su": p.C_su, "S_aa": p.C_aa, "S_fa": p.C_fa, "S_va": p.C_va, "S_bu": p.C_bu, "S_pro": p.C_pro,
         "S_ac": p.C_ac, "S_ch4": p.C_ch4, "S_I": p.C_sI, "X_xc": p.C_xc, "X_ch": p.C_ch, "X_pr": p.C_pr,
         "X_li": p.C_li, "X_I": p.C_xI, **{b: p.C_bac for b in BIOMASS}}
    carbon = sum(C[n] * d[n] for n in C) + d["S_IC"] + gas["Rho_T_10"] + p.C_ch4 * gas["Rho_T_9"]
    # Nitrogen
    N = {"S_aa": p.N_aa, "X_pr": p.N_aa, "S_I": p.N_I, "X_I": p.N_I, "X_xc": p.N_xc, **{b: p.N_bac for b in BIOMASS}}
    nitrogen = sum(N[n] * d[n] for n in N) + d["S_IN"]
    return cod, carbon, nitrogen


def main():
    ok = True
    y_bsm2 = bsm2_state()

    # ---------------------------------------------------------------- 1. use_xc = 1 == validated
    r, p = make_reactor(1)
    y, _ = equilibrated(y_bsm2, p)
    d = np.asarray(r.ADM1_ODE(0.0, y))
    # frozen from the validated commit (same state, same influent row 0)
    ref = {"S_IC": 0.001595585870962625, "S_IN": 0.00042125309676190774, "X_xc": 0.1049810582917647,
           "X_ch": 0.06699711304535294, "S_I": -0.0004236119523529395, "X_I": 0.41936686709411763}
    idx = {n: FULL_STATE_NAMES.index(n) for n in ref}
    worst = max(abs(d[idx[n]] - ref[n]) / max(abs(ref[n]), 1e-30) for n in ref)
    print(f"[1] use_xc=1 vs validated RHS: max rel diff {worst:.1e}  ->", "OK" if worst < 1e-12 else "FAIL")
    ok &= worst < 1e-12

    # ---------------------------------------------------------------- 2. conservation
    y_bio = np.zeros(len(FULL_STATE_NAMES))
    for n, v in {"S_IC": 0.1, "S_IN": 0.1, "S_cation": 0.04, "S_anion": 0.02, "S_h2": 1e-7, "S_ch4": 0.05,
                 "S_gas_h2": 1e-5, "S_gas_ch4": 1.6, "S_gas_co2": 0.014, **{b: 1.0 for b in BIOMASS}}.items():
        y_bio[FULL_STATE_NAMES.index(n)] = v
    for use_xc in (1, 0):
        for label, y in (("BSM2 state", y_bsm2), ("biomass-only state", y_bio)):
            r, p = make_reactor(use_xc)
            cod, c, n = elemental_residuals(r, p, y)
            good = abs(cod) < 1e-12 and abs(c) < 1e-12 and abs(n) < 1e-12
            print(f"[2] use_xc={use_xc} {label:19s} COD {cod:+.1e}  C {c:+.1e}  N {n:+.1e}  ->", "OK" if good else "FAIL")
            ok &= good

    # ---------------------------------------------------------------- 3. bad fractions rejected
    try:
        make_reactor(0, f_ch_xb=0.5)
        print("[3] f_*_xb not summing to 1 accepted -> FAIL"); ok = False
    except ValueError:
        print("[3] f_*_xb not summing to 1 rejected -> OK")

    # ---------------------------------------------------------------- 4. sanity: use_xc=0 => Rho_1 = 0, decay bypasses X_xc
    r, p = make_reactor(0)
    p.params["q_ad"] = 0.0
    y, _ = equilibrated(y_bsm2, p)
    d = dict(zip(FULL_STATE_NAMES, np.asarray(r.ADM1_ODE(0.0, y))))
    good = r._last_rho["Rho_1"] == 0.0 and d["X_xc"] == 0.0
    print(f"[4] use_xc=0: Rho_1 = {r._last_rho['Rho_1']}, dX_xc/dt = {d['X_xc']} ->", "OK" if good else "FAIL")
    ok &= good

    print("\nVERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

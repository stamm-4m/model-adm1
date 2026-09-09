#!/usr/bin/env python3
"""
benchmark_bsm2_steady.py — BSM2 steady-state test (second benchmark).

Protocol (Rosen & Jeppsson 2006; PyADM1 paper Tables 2-4): constant BSM2 digester influent,
q_ad = 170 m3/d, 35 C, 200 d from the published initial state. The final state must equal the
published BSM2 steady state (tests/data/bsm2_steady_state.csv, column BSM2). PyADM1 reaches a
maximum absolute error of 1.9e-5; this implementation reaches ~8e-7 at 200 d and ~4e-11 at 400 d.

Usage:  python main.py            # with active_scenario: BSM2_steady_state
        python tests/benchmark_bsm2_steady.py --results results/dynamic_out.csv [--tol 1e-5]
"""
import argparse, os, sys
import numpy as np, pandas as pd

COD_EQ = {"S_va_ion": 208.0, "S_bu_ion": 160.0, "S_pro_ion": 112.0, "S_ac_ion": 64.0}

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--reference", default=os.path.join(os.path.dirname(__file__), "data", "bsm2_steady_state.csv"))
ap.add_argument("--tol", type=float, default=1e-5, help="max absolute error accepted (PyADM1 level: 1.9e-5)")
a = ap.parse_args()

res = pd.read_csv(a.results)
last = res.iloc[-1]
ref = pd.read_csv(a.reference).set_index("state")
rows = []
for s in ref.index:
    if s not in res.columns:
        continue
    v = float(last[s])
    if s in COD_EQ and abs(v / max(float(last[s.replace("_ion", "")]), 1e-30) - 1 / COD_EQ[s]) < 0.5 / COD_EQ[s]:
        v *= COD_EQ[s]                       # kmol/m3 -> kgCOD/m3 (MATLAB/PyADM1 convention)
    rows.append(dict(state=s, BSM2=ref.BSM2[s], model=v, abs_err=v - ref.BSM2[s],
                     pyadm1_abs_err=ref.PyADM1[s] - ref.BSM2[s], PASS=abs(v - ref.BSM2[s]) <= a.tol))
tab = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"BSM2 steady-state test — final state at t = {last['time']:g} d vs published BSM2 steady state")
print(tab.to_string(index=False, float_format=lambda x: f"{x:.10g}"))
worst = tab.abs_err.abs().max()
n_fail = int((~tab.PASS).sum())
print(f"\nmax |abs err| = {worst:.2e}  (PyADM1 paper: {tab.pyadm1_abs_err.abs().max():.2e}; tolerance {a.tol:g})")
print("VERDICT:", "PASS" if n_fail == 0 else f"FAIL ({n_fail} states)")
outdir = os.path.join(os.path.dirname(a.results), "benchmark_steady")
os.makedirs(outdir, exist_ok=True)
tab.to_csv(os.path.join(outdir, "steady_state_table.csv"), index=False)
sys.exit(0 if n_fail == 0 else 1)

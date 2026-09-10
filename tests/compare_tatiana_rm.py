"""
Cross-test of the use_xc = 0 implementation against the RM_without_Xc code (Tatiana's 5 L digester).

    python main.py                       # active_scenario: Tatiana_RM_lab_5L
    python tests/compare_tatiana_rm.py --results results/dynamic_out.csv

Reference: tests/data/tatiana_rm_sim_result.csv = RM_without_Xc/outputs/sim_result.csv (193 d, 40 rows/d),
produced by the RM notebook with parameters_dyn.txt and its y0. Both runs are sampled at integer days.

Expected agreement: NOT bit-for-bit. The biochemistry is the same (see docs), but the two codes differ in
  - gas flow law (k_p overpressure vs. P_gas = P_atm assumption)
  - acid-base formulation (algebraic charge balance vs. 6 ion ODEs with k_A_B = 1e10) and integrator
  - RM applies no integrator restart at feed steps and clips negative states in place
so the tolerances below are those of a model cross-check, not of a numerical identity.

Known 1-day offset (--rm-shift-days, default 1): the RM notebook reads the influent with
pd.read_csv("U_dyn_ref.csv", sep=";") WITHOUT header=None, so the first data row (day 0) is consumed as
the header and every feed row is applied one day early. Our configs/tatiana_lab_5L_influent.csv keeps the
correct alignment (day 0 = 2023-03-15); the RM trajectory is therefore compared at t + 1 d.
"""
import argparse
import sys

import numpy as np
import pandas as pd

RM_TO_OURS = {"S_su": "S_su", "S_aa": "S_aa", "S_fa": "S_fa", "S_va": "S_va", "S_bu": "S_bu", "S_pro": "S_pro",
              "S_ac": "S_ac", "S_h2": "S_h2", "S_ch4": "S_ch4", "S_IC": "S_IC", "S_IN": "S_IN", "S_I": "S_I",
              "Xxc": "X_xc", "Xch": "X_ch", "Xpr": "X_pr", "Xli": "X_li", "X_su": "X_su", "X_aa": "X_aa",
              "X_fa": "X_fa", "X_c4": "X_c4", "X_pro": "X_pro", "X_ac": "X_ac", "X_h2": "X_h2", "X_I": "X_I",
              "S_cat": "S_cation", "S_an": "S_anion", "S_hco3": "S_hco3_ion", "S_nh3": "S_nh3",
              "S_gas_h2": "S_gas_h2", "S_gas_ch4": "S_gas_ch4", "S_gas_co2": "S_gas_co2"}
KEY = ["S_ac", "S_pro", "S_bu", "S_va", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I", "X_ch", "X_pr", "X_li", "X_su",
       "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion", "S_hco3_ion", "S_nh3",
       "S_gas_h2", "S_gas_ch4", "S_gas_co2", "pH"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/dynamic_out.csv")
    ap.add_argument("--reference", default="tests/data/tatiana_rm_sim_result.csv")
    ap.add_argument("--tol-avg", type=float, default=3.0, help="max |avg err| %% on the 193-d mean")
    ap.add_argument("--tol-rmse", type=float, default=10.0, help="max daily nRMSE %%")
    ap.add_argument("--tol-ph", type=float, default=0.05, help="max |pH avg err| [units]")
    ap.add_argument("--rm-shift-days", type=float, default=1.0,
                    help="RM feed is applied this many days early (header bug in the RM notebook); 0 = no shift")
    a = ap.parse_args()

    ours = pd.read_csv(a.results)
    rm = pd.read_csv(a.reference)
    # RM stores H+ in the dummy state D4 (post_comp writes x[:,40] in place before save)
    rm["pH"] = -np.log10(rm["D4"].clip(lower=1e-14))
    rm = rm.rename(columns=RM_TO_OURS)
    days = np.arange(0, int(min(ours["time"].max(), rm["time"].max() + a.rm_shift_days)) + 1)
    o = ours.set_index("time").reindex(days, method="nearest")
    r = rm.set_index("time").reindex(days - a.rm_shift_days, method="nearest")   # RM(t - shift) <-> ours(t)

    print(f"Tatiana_RM_lab_5L vs RM_without_Xc — {len(days)} daily samples, days 0–{days[-1]} "
          f"(RM shifted by +{a.rm_shift_days:g} d, see docstring)")
    print(f"{'state':12s}{'ours avg':>12s}{'RM avg':>12s}{'avg err %':>11s}{'nRMSE %':>10s}   ok")
    ok_all = True
    rows = []
    for s in KEY:
        if s not in o.columns or s not in r.columns:
            continue
        x, y = o[s].values.astype(float), r[s].values.astype(float)
        if s == "pH":
            err, rmse = x.mean() - y.mean(), np.sqrt(np.mean((x - y) ** 2))
            good = abs(err) <= a.tol_ph
            print(f"{s:12s}{x.mean():12.4f}{y.mean():12.4f}{err:+11.3f}{rmse:10.3f}   {'✓' if good else '✗'}  (units)")
        else:
            scale = max(abs(y.mean()), 1e-30)
            err = 100 * (x.mean() - y.mean()) / scale
            rmse = 100 * np.sqrt(np.mean((x - y) ** 2)) / scale
            good = abs(err) <= a.tol_avg and rmse <= a.tol_rmse
            print(f"{s:12s}{x.mean():12.4e}{y.mean():12.4e}{err:+11.2f}{rmse:10.2f}   {'✓' if good else '✗'}")
        rows.append((s, x.mean(), y.mean(), err, rmse, good))
        ok_all &= good
    pd.DataFrame(rows, columns=["state", "ours_avg", "rm_avg", "avg_err_pct", "nrmse_pct", "ok"]).to_csv(
        "results/tatiana_crosstest.csv", index=False)
    print("\nVERDICT:", "PASS" if ok_all else "FAIL", f"(tol: |avg| ≤ {a.tol_avg} %, nRMSE ≤ {a.tol_rmse} %, pH ≤ {a.tol_ph})")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

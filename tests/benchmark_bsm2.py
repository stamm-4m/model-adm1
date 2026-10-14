#!/usr/bin/env python3
"""
benchmark_bsm2.py — BSM2 ring test for a Python ADM1 implementation.

The benchmark is the one PyADM1 itself was validated against (Sadrimajd et al., 2021):
the ADM1 digester of Benchmark Simulation Model No. 2 (Rosen & Jeppsson, 2006),
280 days of dynamic influent at 15-min resolution, simulated with the reference
MATLAB/Simulink BSM2 implementation. The reference trajectories ship with PyADM1
as  src/Matlabout_dyn.csv  (26 881 rows x 38 states + time).

What this script does
---------------------
  1. loads your simulation output (any time resolution, daily or 15-min)
  2. loads the MATLAB reference
  3. computes, for every state variable, the ring-test metric used by PyADM1
     (time-integrated 280-day average, trapezoidal rule) and the relative error,
     plus a dynamic metric (normalised RMSE of daily means after the start-up
     transient) and the maximum absolute deviation
  4. compares against tolerances and against the PyADM1 yardstick (the errors the
     original PyADM1 code obtains on the same test — a refactor should be at
     least as close to MATLAB as PyADM1 is)
  5. writes a CSV table, a Markdown summary and a PNG overlay plot; exit code 0 = pass

Usage
-----
  python benchmark_bsm2.py --results results/dynamic_out.csv \
                           --reference path/to/Matlabout_dyn.csv \
                           --outdir results/benchmark

  Options:
    --skip-days 30      days excluded from the dynamic (RMSE) metric (start-up transient)
    --tol-avg 6.0       max |relative error| (%) on the 280-day average, all states
                        (PyADM1 itself reaches 5.2 % on X_I because of the constant q_ad)
    --tol-ph 0.05       max |error| on the 280-day average pH (pH units)
    --tol-rmse 25.0     max daily-mean nRMSE (%) — with a constant q_ad the BSM2 VFAs
                        cannot do better than ~17 % (PyADM1 yardstick); pass --dynamic-q
                        if your run used the influent Q column, tolerance is tightened to 12 %

Column conventions handled automatically
----------------------------------------
  * pH may be given as 'pH' or derived from 'S_H_ion'
  * VFA ions (S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion) in kgCOD/m3 (PyADM1/MATLAB
    convention) or in kmol/m3 (refactored acid_base.py convention) — detected from
    the ratio to the total acid and converted to kgCOD/m3.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy import integrate

STATES = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
          "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
          "S_cation", "S_anion", "pH", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion",
          "S_co2", "S_nh3", "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2"]
COD_EQ = {"va": 208.0, "bu": 160.0, "pro": 112.0, "ac": 64.0}
KEY_PLOT = ["pH", "S_ac", "S_pro", "S_h2", "S_IC", "S_IN", "S_nh3", "X_ac", "S_gas_ch4", "S_gas_co2", "S_gas_h2", "S_hco3_ion"]

# PyADM1 (original, constant q_ad=178.47, DOP853, 15-min DAE) errors vs MATLAB on this test,
# re-run Sept 2026 with scipy 1.17 — used as the yardstick column. |avg err %|, daily nRMSE %.
PYADM1_YARDSTICK = {
    "S_su": (1.73, 12.9),
    "S_aa": (1.87, 14.6),
    "S_fa": (2.21, 14.7),
    "S_va": (2.26, 15.1),
    "S_bu": (2.03, 15.1),
    "S_pro": (2.49, 16.0),
    "S_ac": (2.17, 17.0),
    "S_h2": (1.78, 13.3),
    "S_ch4": (0.08, 1.9),
    "S_IC": (0.36, 1.4),
    "S_IN": (0.34, 1.4),
    "S_I": (3.92, 5.9),
    "X_xc": (1.53, 2.1),
    "X_ch": (2.35, 13.8),
    "X_pr": (1.73, 14.5),
    "X_li": (2.27, 14.0),
    "X_su": (3.67, 4.1),
    "X_aa": (0.12, 1.1),
    "X_fa": (3.64, 4.1),
    "X_c4": (0.14, 1.1),
    "X_pro": (1.03, 1.6),
    "X_ac": (1.43, 1.9),
    "X_h2": (1.86, 2.3),
    "X_I": (5.18, 5.3),
    "S_anion": (0.43, 0.7),
    "pH": (0.01, 0.1),
    "S_va_ion": (2.26, 15.1),
    "S_bu_ion": (2.03, 15.1),
    "S_pro_ion": (2.49, 16.0),
    "S_ac_ion": (2.17, 17.0),
    "S_hco3_ion": (0.38, 1.6),
    "S_co2": (0.24, 1.1),
    "S_nh3": (0.36, 3.4),
    "S_nh4_ion": (0.33, 1.3),
    "S_gas_h2": (1.79, 11.7),
    "S_gas_ch4": (0.1, 0.5),
    "S_gas_co2": (0.21, 1.1),
}


def load_results(path, ref_time=None):
    df = pd.read_csv(path)
    if "time" not in df.columns:
        if ref_time is not None and len(df) == len(ref_time):
            df.insert(0, "time", ref_time)   # PyADM1's dynamic_out.csv has no time column: same grid as reference
            print("  note: no 'time' column; adopted the reference time grid (same length)")
        else:
            raise SystemExit("results file needs a 'time' column (days)")
    if "pH" not in df.columns:
        if "S_H_ion" in df.columns:
            df["pH"] = -np.log10(df["S_H_ion"].clip(lower=1e-14))
        else:
            raise SystemExit("results need 'pH' or 'S_H_ion'")
    # unit detection for VFA ions: molar values are ~1/64..1/208 of the COD totals
    for sp, f in COD_EQ.items():
        ion, tot = f"S_{sp}_ion", f"S_{sp}"
        if ion in df.columns and tot in df.columns:
            ratio = (df[ion] / df[tot].replace(0, np.nan)).median()
            if 0.5 / f < ratio < 2.0 / f:
                df[ion] = df[ion] * f
                print(f"  note: {ion} detected in kmol/m3 -> converted to kgCOD/m3 (x{f:g})")
    return df


def time_average(t, y):
    return integrate.trapezoid(y, t) / (t[-1] - t[0])


def daily_means(t, y, day_lo, day_hi):
    day = np.floor(t - 1e-9).astype(int)
    day[0] = int(np.floor(t[0]))
    s = pd.Series(y).groupby(day).mean()
    return s.reindex(range(day_lo, day_hi)).values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--reference", required=True, help="PyADM1/src/Matlabout_dyn.csv")
    ap.add_argument("--outdir", default="benchmark_out")
    ap.add_argument("--skip-days", type=int, default=30)
    ap.add_argument("--tol-avg", type=float, default=6.0)
    ap.add_argument("--tol-ph", type=float, default=0.05)
    ap.add_argument("--tol-rmse", type=float, default=25.0)
    ap.add_argument("--dynamic-q", action="store_true", help="run used the influent Q column: tighten RMSE tolerance to 12 %%")
    ap.add_argument("--label", default="model")
    a = ap.parse_args()
    if a.dynamic_q:
        a.tol_rmse = min(a.tol_rmse, 12.0)
    os.makedirs(a.outdir, exist_ok=True)

    ref = pd.read_csv(a.reference)
    res = load_results(a.results, ref.time.values)
    t_end = min(ref.time.iloc[-1], res.time.iloc[-1])
    if t_end < ref.time.iloc[-1] - 1e-6:
        print(f"  warning: results end at {t_end:g} d, reference at {ref.time.iloc[-1]:g} d; comparing on [0,{t_end:g}]")
    ref = ref[ref.time <= t_end + 1e-9].reset_index(drop=True)
    res = res[res.time <= t_end + 1e-9].reset_index(drop=True)
    if len(res) < 50:
        print("  warning: fewer than 50 output points; use dt_out <= 1 d for a meaningful test")

    d_lo, d_hi = a.skip_days, int(np.floor(t_end))
    rows = []
    for s in STATES:
        if s not in res.columns or s not in ref.columns:
            continue
        m_avg = time_average(ref.time.values, ref[s].values)
        r_avg = time_average(res.time.values, res[s].values)
        near_zero = abs(m_avg) < 1e-8          # e.g. S_cation is ~0 in the BSM2 digester: relative error meaningless
        scale = abs(m_avg) if not near_zero else 1.0
        err_avg = (r_avg - m_avg) / scale * 100.0
        dm = daily_means(ref.time.values, ref[s].values, d_lo, d_hi)
        dr = daily_means(res.time.values, res[s].values, d_lo, d_hi)
        ok = ~(np.isnan(dm) | np.isnan(dr))
        nrmse = np.sqrt(np.mean((dr[ok] - dm[ok]) ** 2)) / scale * 100.0 if ok.any() else np.nan
        maxdev = np.nanmax(np.abs(dr[ok] - dm[ok])) if ok.any() else np.nan
        if near_zero:
            pass_avg = abs(r_avg - m_avg) <= 1e-4      # absolute: 0.1 mmol/L
        elif s == "pH":
            pass_avg = abs(r_avg - m_avg) <= a.tol_ph
        else:
            pass_avg = abs(err_avg) <= a.tol_avg
        pass_rmse = (nrmse <= a.tol_rmse) if (not np.isnan(nrmse) and not near_zero) else True
        if near_zero:
            err_avg, nrmse = (r_avg - m_avg), np.nan   # report absolute deviation instead
        y = PYADM1_YARDSTICK.get(s)
        rows.append(dict(state=s, matlab_avg=m_avg, model_avg=r_avg, avg_err_pct=err_avg,
                         daily_nRMSE_pct=nrmse, max_daily_dev=maxdev,
                         pyadm1_avg_err_pct=(y[0] if y else np.nan), pyadm1_nRMSE_pct=(y[1] if y else np.nan),
                         PASS=bool(pass_avg and pass_rmse)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.outdir, "benchmark_table.csv"), index=False)

    n_fail = int((~tab.PASS).sum())
    verdict = "PASS" if n_fail == 0 else f"FAIL ({n_fail} state(s) outside tolerance)"
    pd.set_option("display.width", 200)
    print(f"\nBSM2 ring test — {a.label} vs MATLAB BSM2 on [0, {t_end:g}] d   (RMSE on daily means, days {d_lo}-{d_hi})")
    print(tab.round(4).to_string(index=False))
    print(f"\nTolerances: |avg err| <= {a.tol_avg} % (pH: {a.tol_ph} units), daily nRMSE <= {a.tol_rmse} %")
    print(f"VERDICT: {verdict}")

    with open(os.path.join(a.outdir, "benchmark_summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# BSM2 ring test — {a.label}\n\nReference: MATLAB BSM2 ADM1 (Rosen & Jeppsson 2006), file `{os.path.basename(a.reference)}`.\n")
        f.write(f"Horizon 0–{t_end:g} d; RMSE on daily means, days {d_lo}–{d_hi}.\n\n")
        f.write(f"**Verdict: {verdict}**\n\nTolerances: |avg err| ≤ {a.tol_avg} % (pH ≤ {a.tol_ph}), daily nRMSE ≤ {a.tol_rmse} %.\n\n")
        f.write(tab.round(4).to_markdown(index=False))
        f.write("\n\n`pyadm1_*` columns = what the original PyADM1 obtains on the same test (yardstick).\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [k for k in KEY_PLOT if k in tab.state.values]
        fig, axes = plt.subplots(int(np.ceil(len(keys) / 3)), 3, figsize=(15, 3.2 * np.ceil(len(keys) / 3)), sharex=True)
        for ax, s in zip(axes.flat, keys):
            ax.plot(ref.time, ref[s], color="#c44", lw=0.6, alpha=0.7, label="MATLAB BSM2")
            ax.plot(res.time, res[s], color="#2266cc", lw=0.9, label=a.label)
            r = tab[tab.state == s].iloc[0]
            ax.set_title(f"{s}   avg err {r.avg_err_pct:+.2f} %   nRMSE {r.daily_nRMSE_pct:.1f} %", fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes.flat[len(keys):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("time [d]")
        fig.suptitle(f"BSM2 ring test — {a.label}: {verdict}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, "benchmark_overlay.png"), dpi=110)
        print(f"Outputs in {a.outdir}/: benchmark_table.csv, benchmark_summary.md, benchmark_overlay.png")
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

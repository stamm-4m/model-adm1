"""
Influent-hold study (backlog 63, 2026-11-06): which influent row should be held over each interval
[t_(i-1), t_i]? Runs the BSM2 ring test with each convention of simulation.yaml influent_hold
(start = row i-1, end = row i as the PyADM1 loop, legacy = model-adm1 before 2026-11-06) and scores each run
against the MATLAB/Simulink BSM2 reference with tests/benchmark_bsm2.py (--dynamic-q).

    python tools/influent_hold_study.py                  # BSM2_ringtest (daily influent, ~1 min a run)
    python tools/influent_hold_study.py --with-15min     # + BSM2_ringtest_15min (~25 min a run)

The runs go in parallel to a temporary folder. Writes tools/influent_hold_report.md (summary
table + the benchmark table of every run); the benchmark folders are kept in the temporary folder.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, "tests", "data", "Matlabout_dyn.csv")
HOLDS = ("legacy", "start", "end")
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run(cmd, log):
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True, env=ENV).returncode
    return rc, time.time() - t0


def one(scenario, hold, base):
    tag = f"{scenario}_{hold}"
    runs = os.path.join(base, f"runs_{tag}")
    rc, secs = run([sys.executable, "main.py", "--scenario", scenario, "--influent-hold", hold, "--runs-dir", runs,
                    "--no-results", "--no-plots"], os.path.join(base, f"{tag}_run.log"))
    csvs = sorted(glob.glob(os.path.join(runs, "*", "dynamic_out.csv")))
    if rc != 0 or not csvs:
        return {"scenario": scenario, "hold": hold, "error": f"run failed (exit {rc}), see {tag}_run.log"}
    out = os.path.join(base, f"bench_{tag}")
    rc_b, _ = run([sys.executable, os.path.join("tests", "benchmark_bsm2.py"), "--results", csvs[-1], "--reference",
                   REFERENCE, "--outdir", out, "--dynamic-q", "--label", f"{scenario} ({hold})"],
                  os.path.join(base, f"{tag}_bench.log"))
    table = os.path.join(out, "benchmark_table.csv")
    if not os.path.exists(table):
        return {"scenario": scenario, "hold": hold, "error": f"benchmark failed (exit {rc_b}), see {tag}_bench.log"}
    tab = pd.read_csv(table)
    states = tab[tab["state"] != "pH"]
    ph = tab[tab["state"] == "pH"].iloc[0]
    return {"scenario": scenario, "hold": hold, "run_s": round(secs), "table": tab,
            "max_abs_avg_err_pct": float(states["avg_err_pct"].abs().max()),
            "mean_abs_avg_err_pct": float(states["avg_err_pct"].abs().mean()),
            "max_daily_nRMSE_pct": float(states["daily_nRMSE_pct"].max(skipna=True)),
            "mean_daily_nRMSE_pct": float(states["daily_nRMSE_pct"].mean(skipna=True)),
            "pH_avg_err": float(ph["model_avg"] - ph["matlab_avg"]),
            "fails": int((~tab["PASS"].astype(bool)).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-15min", action="store_true")
    a = ap.parse_args(argv)
    scenarios = ["BSM2_ringtest"] + (["BSM2_ringtest_15min"] if a.with_15min else [])
    base = tempfile.mkdtemp(prefix="adm1_hold_")
    jobs = [(s, h) for s in scenarios for h in HOLDS]
    with ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda j: one(*j, base), jobs))
    lines = [f"# Influent-hold study — {time.strftime('%Y-%m-%d %H:%M')}", "",
             "BSM2 ring test against the MATLAB/Simulink reference (tests/data/Matlabout_dyn.csv), "
             "tests/benchmark_bsm2.py --dynamic-q (avg = 280-day time average; nRMSE of daily means, days 30-280; "
             "pH error in pH units; the 38 states and the acid-base species except pH).", "",
             "| scenario | hold | max abs avg err % | mean abs avg err % | max daily nRMSE % | mean daily nRMSE % "
             "| pH avg err | states failing | run s |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        if "error" in r:
            lines.append(f"| {r['scenario']} | {r['hold']} | ERROR: {r['error']} | | | | | | |")
        else:
            lines.append(f"| {r['scenario']} | {r['hold']} | {r['max_abs_avg_err_pct']:.3f} | {r['mean_abs_avg_err_pct']:.3f} "
                         f"| {r['max_daily_nRMSE_pct']:.3f} | {r['mean_daily_nRMSE_pct']:.3f} | {r['pH_avg_err']:+.4f} "
                         f"| {r['fails']} | {r['run_s']} |")
    for r in res:
        if "table" in r:
            lines += ["", f"## {r['scenario']}, hold {r['hold']}", "",
                      r["table"][["state", "matlab_avg", "model_avg", "avg_err_pct", "daily_nRMSE_pct", "PASS"]]
                      .to_markdown(index=False, floatfmt=".4f")]
    lines += ["", f"Temporary folder (runs, benchmark outputs, logs): {base}"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "tools", "influent_hold_report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print("\n".join(lines[:5 + len(res) + 1]))
    print(f"\nFull report: tools/influent_hold_report.md   (temporary folder: {base})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

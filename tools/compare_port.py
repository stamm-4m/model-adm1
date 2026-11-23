"""
Port check (backlog 15, step 3, 2026-11-06): the previous code (an untouched copy of model-adm1,
STAMM/backups/model_adm1_original_20261106, copied to a temporary folder) and the code on
stamm-model-core (this folder) run the same scenarios; for each one the report says whether

  * dynamic_out.csv is identical (same header, same values written; line endings ignored: the
    previous code wrote it with pandas, CRLF on Windows), else the largest difference per column,
  * the parameter values of run.json are the same (the previous package also listed T_ad, an
    alias of T_op that no equation reads),
  * the new package is accepted by the registry's own parser (model-registry, next to this folder).

    python tools/compare_port.py                     # BSM2 and hybrid scenarios (not Tatiana's)
    python tools/compare_port.py --only BSM2_ringtest,BSM2_steady_state
    python tools/compare_port.py --with-15min        # + BSM2_ringtest_15min (~50 min per code)

The cases run in parallel (--jobs). Writes tools/compare_port_report.txt; nothing else in this
folder (runs and results go to temporary folders).
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_SRC = os.path.join(ROOT, "..", "backups", "model_adm1_original_20261106")
REGISTRY_PARSER = os.path.join(ROOT, "..", "model-registry", "model_registry", "api", "services",
                               "dynamic_model_package.py")
CASES = ["BSM2_ringtest", "BSM2_dynamic", "BSM2_steady_state", "BSM2_constant", "thermophilic", "batch_validation",
         "pig_slurry_test", "no_xc_demo", "hybrid_demo", "hybrid_lr_demo", "hybrid_lr_spec_demo"]
SLOW = ["BSM2_ringtest_15min"]


def registry():
    if not os.path.exists(REGISTRY_PARSER):
        return None
    spec = importlib.util.spec_from_file_location("registry_dynamic_model_package", REGISTRY_PARSER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(cmd, cwd, log):
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        # UTF-8 output: the previous main.py prints box-drawing characters (cp1252 fails on them)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
    with open(log, encoding="utf-8", errors="replace") as fh:
        return p.returncode, fh.read(), time.time() - t0


def lines(blob: bytes) -> list:
    return blob.decode("utf-8").replace("\r\n", "\n").rstrip("\n").split("\n")


def csv_diff(a: bytes, b: bytes) -> str:
    la, lb = lines(a), lines(b)
    if la == lb:
        return "identical" + ("" if a == b else " (line endings differ)")
    ra, rb = list(csv.reader(io.StringIO("\n".join(la)))), list(csv.reader(io.StringIO("\n".join(lb))))
    if ra[0] != rb[0]:
        return f"DIFFERENT columns: only previous {sorted(set(ra[0]) - set(rb[0]))}, only new {sorted(set(rb[0]) - set(ra[0]))}"
    if len(ra) != len(rb):
        return f"DIFFERENT rows: {len(ra) - 1} vs {len(rb) - 1}"
    worst, n = {}, 0
    for x, y in zip(ra[1:], rb[1:]):
        for name, u, v in zip(ra[0], x, y):
            if u != v:
                n += 1
                try:
                    d = abs(float(u) - float(v)) if u and v else float("inf")
                except ValueError:
                    d = float("inf")
                worst[name] = max(worst.get(name, 0.0), d)
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:8]
    return f"DIFFERENT values in {n} cells, largest abs. difference: " + ", ".join(f"{k} {v:.3g}" for k, v in top)


def legacy_copy(base, key):
    d = os.path.join(base, f"legacy_{key}")
    shutil.copytree(LEGACY_SRC, d, ignore=shutil.ignore_patterns("runs", "results", "__pycache__", "tests", "docs", "plots_out"))
    p = os.path.join(d, "configs", "scenario.yaml")
    with open(p, encoding="utf-8", newline="") as fh:
        s = fh.read()
    s, n = re.subn(r"(?m)^active_scenario:.*$", f"active_scenario: {key}", s)
    if n != 1:
        raise SystemExit(f"legacy scenario.yaml: expected one active_scenario line, found {n}")
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(s)
    return d


def one_case(key, base):
    legacy = legacy_copy(base, key)
    new_runs = os.path.join(base, f"new_runs_{key}")
    jobs = {
        "old": ([sys.executable, "main.py"], legacy, os.path.join(base, f"{key}_old.log")),
        # the previous code's influent convention (backlog 63): the port itself must change nothing
        "new": ([sys.executable, "main.py", "--scenario", key, "--influent-hold", "legacy", "--runs-dir", new_runs,
                 "--no-results", "--no-plots"],
                ROOT, os.path.join(base, f"{key}_new.log")),
    }
    with ThreadPoolExecutor(2) as ex:
        futs = {k: ex.submit(run, *v) for k, v in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    z_old = sorted(glob.glob(os.path.join(legacy, "runs", "*.zip")))
    z_new = sorted(glob.glob(os.path.join(new_runs, "*.zip")))
    return key, out, (z_old[-1] if z_old else None), (z_new[-1] if z_new else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated scenario keys")
    ap.add_argument("--with-15min", action="store_true", help="also BSM2_ringtest_15min (~50 min per code)")
    ap.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)),
                    help="scenarios run at the same time (each runs the previous and the new code)")
    a = ap.parse_args(argv)
    if not os.path.isdir(LEGACY_SRC):
        raise SystemExit(f"previous code not found: {LEGACY_SRC}")
    cases = a.only.split(",") if a.only else CASES + (SLOW if a.with_15min else [])
    reg = registry()
    base = tempfile.mkdtemp(prefix="adm1_port_")
    lines_out = [f"Port check {time.strftime('%Y-%m-%d %H:%M')}: previous code = {os.path.normpath(LEGACY_SRC)}, "
                 f"new code = this folder; registry parser: {'found' if reg else 'NOT FOUND'}; jobs {a.jobs}", ""]
    all_ok = True
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda k: one_case(k, base), cases))
    for key, out, z_o, z_n in results:
        (rc_o, log_o, s_o), (rc_n, log_n, s_n) = out["old"], out["new"]
        lines_out.append(f"=== {key}  (previous {s_o:.0f} s, new {s_n:.0f} s)")
        if not z_o:
            lines_out += [f"  previous code FAILED (exit {rc_o}):", "  " + log_o[-800:].replace("\n", "\n  ")]
            all_ok = False
            continue
        if not z_n:
            lines_out += [f"  new code FAILED (exit {rc_n}):", "  " + log_n[-1500:].replace("\n", "\n  ")]
            all_ok = False
            continue
        with zipfile.ZipFile(z_o) as zo, zipfile.ZipFile(z_n) as zn:
            d = csv_diff(zo.read("dynamic_out.csv"), zn.read("dynamic_out.csv"))
            all_ok &= d.startswith("identical")
            lines_out.append(f"  dynamic_out.csv: {d}")
            if "influent.csv" in zo.namelist():
                same = "influent.csv" in zn.namelist() and zo.read("influent.csv") == zn.read("influent.csv")
                all_ok &= same
                lines_out.append(f"  influent.csv: {'identical' if same else 'DIFFERENT or missing'}")
            jo, jn = json.loads(zo.read("run.json")), json.loads(zn.read("run.json"))
        po = {p["name"]: p["value"] for p in jo["calibration"]["parameters"] if p["name"] != "T_ad"}
        pn = {p["name"]: p["value"] for p in jn["calibration"]["parameters"]}
        diff = sorted(k for k in set(po) | set(pn) if po.get(k) != pn.get(k))
        all_ok &= not diff
        lines_out.append("  parameters: " + ("same values" if not diff else
                                             "DIFFERENT " + ", ".join(f"{k} {po.get(k)} vs {pn.get(k)}" for k in diff)))
        computed = [p["name"] for p in jn["calibration"]["parameters"] if p.get("source") == "computed"]
        lines_out.append(f"  computed by the model: {computed}; scenario fingerprint (new) "
                         f"{reg.scenario_fingerprint(jn['calibration'])[:12] if reg else '-'}")
        if reg:
            try:
                with open(z_n, "rb") as fh:
                    pk = reg.parse_package(fh.read())
                lines_out.append(f"  registry parser: ACCEPTED ({pk.n_rows} rows, scenario {pk.scenario['name']})")
            except reg.PackageError as exc:
                all_ok = False
                lines_out.append(f"  registry parser: REFUSED {exc.problems}")
        lines_out.append(f"  warnings: previous {jo['run']['status']['warnings']}, new {jn['run']['status']['warnings']}")
    lines_out += ["", "RESULT: " + ("all identical and accepted" if all_ok else "differences found (see above)"),
                  f"temporary folder (logs): {base}"]
    text = "\n".join(lines_out)
    print(text)
    with open(os.path.join(ROOT, "tools", "compare_port_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

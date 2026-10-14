"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Module: run_export — standard run package for the STAMM model registry

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 14/10/2026

Every simulation run produces a self-describing package (see docs/run_package.md):

    runs/<run_id>/
    ├── dynamic_out.csv   simulation trajectories (same content as results/dynamic_out.csv)
    ├── run.json          model / calibration / run / outputs / reference metadata
    └── influent.csv      copy of the influent (dynamic influent only)
    runs/<run_id>.zip     the same three files, ready to upload to STAMM

run.json sections map onto the three registry levels:
    model        -> family     (the code)
    calibration  -> instance   (resolved parameters, initial states, influent, hybrid hooks)
    run/outputs  -> simulation (one execution + description of every CSV column)

The CSV itself is never modified: this module only adds files next to it.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import yaml


SCHEMA_VERSION = "1.0"

DEFAULT_PATHS = {
    "params": "configs/adm1_parameters.yaml",
    "states": "configs/initial_states.yaml",
    "influent": "configs/influent.yaml",
    "scenario": "configs/scenario.yaml",
    "simulation": "configs/simulation.yaml",
    "model_card": "configs/model_card.yaml",
    "output_variables": "configs/output_variables.yaml",
}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _clean(obj):
    """Make an object JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, str) or obj is None:
        return obj
    return str(obj)


def _text(value):
    """Collapse YAML folded/multi-line text into one line."""
    if value is None:
        return None
    return " ".join(str(value).split())


def _same(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    except (TypeError, ValueError):
        return a == b


def _same_units(a, b) -> bool:
    """Unit strings that mean the same thing: M = kmol.m^-3, and 'kmol C.m^-3' = 'kmol.m^-3'
    (the C / N element label is informative only). kgCOD vs kmol stays a real difference."""
    def norm(u):
        if u is None:
            return None
        u = str(u).replace(" ", "")
        u = re.sub(r"^kmol[CN]\.", "kmol.", u)
        return "kmol.m^-3" if u == "M" else u
    return norm(a) == norm(b)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "run"


# ----------------------------------------------------------------------
# model  (family)
# ----------------------------------------------------------------------

def code_reference(root: str = ".") -> dict:
    """
    Identify the exact code that produced the run:
      - source_sha256 : hash of main.py, initial_states.py and src/*.py (always available)
      - git_commit / git_branch / uncommitted_changes : read-only git queries, None if no git
    """
    files = ["main.py", "initial_states.py"] + sorted(
        os.path.relpath(p, root).replace("\\", "/") for p in glob.glob(os.path.join(root, "src", "*.py"))
    )
    files = [f for f in files if os.path.exists(os.path.join(root, f))]
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode("utf-8"))
        with open(os.path.join(root, rel), "rb") as f:
            h.update(f.read())

    def _git(*args):
        try:
            out = subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, timeout=10,
            )
            return out
        except Exception:
            return None

    commit = branch = dirty = None
    res = _git("rev-parse", "HEAD")
    if res is not None and res.returncode == 0:
        commit = res.stdout.strip()
        res_b = _git("rev-parse", "--abbrev-ref", "HEAD")
        branch = res_b.stdout.strip() if res_b is not None and res_b.returncode == 0 else None
        # read-only: exit code 1 means the source files differ from the commit
        res_d = _git("diff", "--quiet", "HEAD", "--", *files)
        if res_d is not None and res_d.returncode in (0, 1):
            dirty = res_d.returncode == 1

    return {
        "git_commit": commit,
        "git_branch": branch,
        "uncommitted_changes": dirty,
        "source_sha256": h.hexdigest(),
        "source_files": files,
    }


def build_model_section(model_card_file: str, root: str = ".") -> dict:
    card = _load_yaml(model_card_file)
    section = {
        "name": card.get("name", "ADM1"),
        "implementation": card.get("implementation", "model-adm1"),
        "version": card.get("version"),
        "model_kind": card.get("model_kind", "dynamic"),
        "description": _text(card.get("description")),
        "time_units": card.get("time_units", "d"),
        "authors": card.get("authors", []),
        "references": card.get("references", []),
        "repository": card.get("repository"),
        "license": card.get("license"),
    }
    section["code_ref"] = code_reference(root)
    return section


# ----------------------------------------------------------------------
# calibration  (instance)
# ----------------------------------------------------------------------

def collect_parameters(params_file: str, scenario_block: dict, effective: dict) -> list:
    """
    One entry per parameter, after merging adm1_parameters.yaml with the scenario overrides.

    source:
      "default"  : value of the base YAML (literature)
      "scenario" : overridden by the active scenario (parameter_overrides or T_op)
      "computed" : changed by the model code before integration (e.g. K_w, K_a_co2, K_a_IN
                   recomputed from T_op in ADM1Reactor); default_value keeps the YAML value
    `effective` must be a snapshot of param.params taken BEFORE integration.
    """
    raw = _load_yaml(params_file)
    overrides = dict(scenario_block.get("parameter_overrides") or {})
    if isinstance(scenario_block.get("T_op"), dict):
        overrides.setdefault("T_op", scenario_block["T_op"])

    rows, seen = [], set()
    for group, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        for name, spec in entries.items():
            if not (isinstance(spec, dict) and "value" in spec):
                continue
            base = float(spec["value"])
            value = effective.get(name, base)
            row = {
                "name": name,
                "group": group,
                "value": value,
                "units": spec.get("units"),
                "description": _text(spec.get("description")),
            }
            if name in overrides:
                ovr = overrides[name] if isinstance(overrides[name], dict) else {}
                row["source"] = "scenario"
                row["default_value"] = base
                if ovr.get("description"):
                    row["override_note"] = _text(ovr["description"])
            elif not _same(value, base):
                row["source"] = "computed"
                row["default_value"] = base
                row["override_note"] = "recomputed by the model code before integration"
            else:
                row["source"] = "default"
            rows.append(row)
            seen.add(name)

    # overrides that do not exist in the base YAML
    for name, ovr in overrides.items():
        if name in seen:
            continue
        ovr = ovr if isinstance(ovr, dict) else {"value": ovr}
        rows.append({
            "name": name,
            "group": "scenario_only",
            "value": effective.get(name, ovr.get("value")),
            "units": ovr.get("units"),
            "description": _text(ovr.get("description")),
            "source": "scenario",
            "default_value": None,
        })
        seen.add(name)

    # anything else the code added to the parameter set (e.g. the T_ad alias of T_op)
    for name, value in effective.items():
        if name in seen:
            continue
        rows.append({
            "name": name,
            "group": "derived",
            "value": value,
            "units": None,
            "description": "Alias of T_op" if name == "T_ad" else None,
            "source": "computed",
            "default_value": None,
        })
    return rows


def collect_state_variables(
    state_names: list,
    declared: dict,
    effective_vector,
    dictionary: dict,
    states_file: str,
    states_key: str,
) -> list:
    """
    The 38 ADM1 states with
      initial          : EFFECTIVE value used by the solver (after the acid-base re-projection)
      declared_initial : value declared in initial_states.yaml (S_H_ion / S_anion derived from pH
                         when the set gives a pH)
    """
    declared_units = {}
    block = _load_yaml(states_file).get(states_key, {}) if states_key else {}
    for k, v in (block or {}).items():
        if isinstance(v, dict) and "units" in v:
            declared_units[k] = v["units"]

    eff = np.asarray(effective_vector, dtype=float)
    rows = []
    for i, name in enumerate(state_names):
        d = dictionary.get(name, {}) or {}
        row = {
            "symbol": name,
            "name": d.get("name", name),
            "kind": d.get("kind", "state"),
            "units": d.get("units"),
            "description": d.get("description"),
            "initial": float(eff[i]),
            "declared_initial": declared.get(name),
        }
        du = declared_units.get(name)
        if du is not None and not _same_units(du, row["units"]):
            row["declared_units"] = du
        rows.append(row)
    if isinstance(block, dict) and isinstance(block.get("pH"), dict):
        rows.append({
            "symbol": "pH",
            "name": "Initial pH (declared)",
            "kind": "declared_only",
            "units": "-",
            "description": "pH declared in the initial-state set; S_H_ion and S_anion are derived from it",
            "initial": None,
            "declared_initial": block["pH"].get("value"),
        })
    return rows


def collect_influent(influent, influent_file: str, mode_key: str, run_dir: str) -> dict:
    """Describe the influent. A dynamic influent CSV is copied as influent.csv into run_dir."""
    cfg = _load_yaml(influent_file).get(mode_key, {}) or {}
    mode = getattr(influent, "mode", cfg.get("type"))
    info = {
        "mode_key": mode_key,
        "mode": mode,
        "description": _text(cfg.get("description")),
    }
    if mode == "dynamic":
        src = cfg.get("file_path")
        info.update({
            "file": "influent.csv",
            "source_path": src,
            "time_column": cfg.get("time_column", "time"),
            "flow_column": cfg.get("flow_column"),
            "time_units": cfg.get("time_units", "d"),
        })
        if src and os.path.exists(src):
            dst = os.path.join(run_dir, "influent.csv")
            shutil.copyfile(src, dst)
            info["sha256"] = sha256_file(dst)
            t = np.asarray(influent.get_time(), dtype=float)
            info["n_rows"] = int(len(t))
            info["t_first"] = float(t[0]) if len(t) else None
            info["t_last"] = float(t[-1]) if len(t) else None
        else:
            info["file"] = None
            info["warning"] = f"influent file not found: {src}"
    else:
        declared_units = {k: v.get("units") for k, v in cfg.items() if isinstance(v, dict)}
        values = []
        for name, value in (getattr(influent, "_constant_values", {}) or {}).items():
            values.append({"name": name, "value": value, "units": declared_units.get(name)})
        if isinstance(cfg.get("pH"), dict):
            values.append({"name": "pH", "value": cfg["pH"].get("value"), "units": "-"})
        info["values"] = values
    return info


# ----------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------

def describe_columns(columns, dictionary: dict) -> tuple[list, list]:
    variables, unknown = [], []
    for col in columns:
        d = dictionary.get(col)
        if d is None:
            unknown.append(col)
            d = {"kind": "unknown"}
        variables.append({
            "name": col,
            "label": d.get("name", col),
            "units": d.get("units"),
            "kind": d.get("kind"),
            "description": d.get("description"),
        })
    return variables, unknown


def _environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("numpy", "scipy", "pandas", "yaml", "matplotlib"):
        m = sys.modules.get(mod)
        env[mod] = getattr(m, "__version__", None) if m is not None else None
    return env


def _user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:
        return None


# ----------------------------------------------------------------------
# Entry point called by main.py
# ----------------------------------------------------------------------

def export_run_package(
    *,
    df: pd.DataFrame,
    scenario_name: str,
    scenario_block: dict,
    param_snapshot: dict,
    declared_initial: dict,
    effective_initial,
    state_names: list,
    states_key: str,
    influent,
    hybrid_cfg: dict,
    hybrid_summary: dict,
    solver: dict,
    time_info: dict,
    status: dict,
    started_at: float,
    package_cfg: dict | None = None,
    csv_filename: str = "dynamic_out.csv",
    paths: dict | None = None,
    root: str = ".",
) -> str:
    """Write runs/<run_id>/ (+ runs/<run_id>.zip) and return the run directory."""
    package_cfg = package_cfg or {}
    p = dict(DEFAULT_PATHS)
    p.update(paths or {})

    dictionary = _load_yaml(p["output_variables"])
    influent_mode_key = scenario_block.get("influent_mode")

    # --- run id + folder ---
    started = _dt.datetime.fromtimestamp(started_at).astimezone()
    fingerprint = hashlib.sha256(
        json.dumps(_clean({"s": scenario_name, "p": param_snapshot, "y0": list(effective_initial),
                           "t": started.isoformat()}), sort_keys=True).encode("utf-8")
    ).hexdigest()[:6]
    run_id = f"{started.strftime('%Y%m%dT%H%M%S')}_{_slug(scenario_name)}_{fingerprint}"
    runs_dir = package_cfg.get("runs_dir", "runs")
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)

    # --- simulation CSV: identical call to the legacy export -> identical content ---
    csv_path = os.path.join(run_dir, csv_filename)
    df.to_csv(csv_path, index=False)

    variables, unknown = describe_columns(df.columns, dictionary)
    warnings = list(status.get("warnings", []))
    if unknown:
        warnings.append(f"columns without an entry in output_variables.yaml: {unknown}")

    # --- run.json ---
    config_files = {}
    for key in ("params", "states", "influent", "scenario", "simulation"):
        if os.path.exists(p[key]):
            config_files[os.path.basename(p[key])] = sha256_file(p[key])

    run_json = {
        "schema_version": SCHEMA_VERSION,
        "model": build_model_section(p["model_card"], root),
        "calibration": {
            "scenario": {
                "name": scenario_name,
                "description": _text(scenario_block.get("description")),
                "initial_states": states_key,
                "influent_mode": influent_mode_key,
                "t_end": scenario_block.get("t_end"),
            },
            "parameters": collect_parameters(p["params"], scenario_block, param_snapshot),
            "state_variables": collect_state_variables(
                state_names, declared_initial, effective_initial, dictionary, p["states"], states_key,
            ),
            "influent": collect_influent(influent, p["influent"], influent_mode_key, run_dir),
            "hybrid": {
                "config": hybrid_cfg or {},
                "enabled": bool((hybrid_summary or {}).get("enabled", False)),
                "rate_overrides": (hybrid_summary or {}).get("rate_overrides", []),
                "inhibition_overrides": (hybrid_summary or {}).get("inhibition_overrides", []),
                "residual_correction": (hybrid_summary or {}).get("residual_correction", []),
            },
            "config_files_sha256": config_files,
        },
        "run": {
            "run_id": run_id,
            "created_at": started.isoformat(timespec="seconds"),
            "user": _user(),
            "duration_s": round(_dt.datetime.now().timestamp() - started_at, 3),
            "solver": solver,
            "time": time_info,
            "status": {
                "success": bool(status.get("success", True)),
                "message": status.get("message"),
                "warnings": warnings,
            },
            "environment": _environment(),
        },
        "outputs": {
            "file": csv_filename,
            "sha256": sha256_file(csv_path),
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "time_column": "time",
            "variables": variables,
        },
        "reference": scenario_block.get("reference"),
    }

    json_path = os.path.join(run_dir, "run.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean(run_json), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    # --- zip (flat: the files sit at the root of the archive) ---
    if bool(package_cfg.get("zip", True)):
        zip_path = os.path.join(runs_dir, f"{run_id}.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(os.listdir(run_dir)):
                zf.write(os.path.join(run_dir, name), arcname=name)

    return run_dir

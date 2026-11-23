"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Entry point: run one scenario with stamm-model-core, write results/ and the STAMM run package
(runs/<run_id>.zip), checked with the registry's rules.

    python main.py                                  # default scenario of configs/scenarios.yaml
    python main.py --scenario BSM2_steady_state
    python main.py --list-scenarios | --list-models | --list-hooks

Authors:
    - Margaux Bonal           <margaux.bonal@inrae.fr>
    - David Camilo Corrales   <David-Camilo.Corrales-Munoz@inrae.fr>

Configuration (configs/):
    parameters.yaml      intrinsic ADM1 parameters (kinetic, stoichiometric, physico-chemical, reactor)
    initial_states.yaml  named initial-state sets (a pH sets S_anion)
    influent.yaml        influent modes (dynamic CSV or constant values)
    scenarios.yaml       scenarios: initial set, influent mode, horizon, parameter overrides, hybrid
    simulation.yaml      solver, horizon and output settings
    states.yaml / outputs.yaml / model_card.yaml   the stamm-model-core description of the model
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                      # the configuration and the hybrid registry use relative paths

from stamm_model import StammModelError, apply_overrides, load_config, run_scenario  # noqa: E402

from initial_states import InitialState  # noqa: E402
from model import ADM1  # noqa: E402
from src.influent import Influent  # noqa: E402
from src.parameters import ADM1Parameters  # noqa: E402
from src.registry import format_hooks, format_registry, load_registry  # noqa: E402

STATES_FILE = "configs/initial_states.yaml"
INFLUENT_FILE = "configs/influent.yaml"
SCENARIOS_FILE = "configs/scenarios.yaml"
EXPLICIT_SOLVERS = ("RK45", "RK23", "DOP853")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def output_times(t_start, t_end, dt_out):
    """t_start, t_start + dt_out, ... and t_end itself when it falls between two steps."""
    if dt_out <= 0:
        raise ValueError("dt_out must be strictly positive.")
    if t_end <= t_start:
        raise ValueError("t_end must be strictly greater than t_start.")
    n = int(np.floor((t_end - t_start) / dt_out)) + 1
    t = t_start + np.arange(n) * dt_out
    return np.append(t, t_end) if t[-1] < t_end else t


def influent_record(influent, mode_key: str) -> tuple[dict, dict]:
    """calibration.inputs of run.json (what defines the influent) and the file to put in the package."""
    import yaml
    with open(INFLUENT_FILE, encoding="utf-8") as fh:
        cfg = (yaml.safe_load(fh) or {}).get(mode_key, {}) or {}
    desc = cfg.get("description")
    rec = {"kind": "influent", "mode_key": mode_key, "mode": influent.mode,
           "description": " ".join(str(desc).split()) if desc else None}
    files = {}
    if influent.mode == "dynamic":
        src = cfg.get("file_path")
        t = np.asarray(influent.get_time(), dtype=float)
        rec.update({"influent_file": "influent.csv", "source_path": src, "time_column": cfg.get("time_column", "time"),
                    "flow_column": cfg.get("flow_column"), "time_units": cfg.get("time_units", "d"),
                    "sha256": _sha256(src), "n_rows": int(len(t)), "t_first": float(t[0]), "t_last": float(t[-1])})
        files["influent.csv"] = os.path.join(ROOT, src)
    else:
        units = {k: v.get("units") for k, v in cfg.items() if isinstance(v, dict)}
        rec["values"] = [{"name": k, "value": v, "units": units.get(k)}
                         for k, v in (influent._constant_values or {}).items()]
        if isinstance(cfg.get("pH"), dict):
            rec["values"].append({"name": "pH", "value": cfg["pH"].get("value"), "units": "-"})
    return rec, files


def hybrid_record(summary: dict, hybrid_cfg: dict, registry: dict) -> list:
    """calibration.hybrid: the models of models/*.yaml plugged in, with the sha256 of their files."""
    if not summary.get("enabled"):
        return []
    models = []
    for name in hybrid_cfg.get("use") or []:
        e = registry[name]
        files = {}
        if e.source_path is not None:
            files[str(e.source_path).replace(os.sep, "/")] = _sha256(e.source_path)
        art = str(e.artefact).split(":")[0]
        for cand in (os.path.join("models", art), art):
            if os.path.isfile(cand):
                files[cand.replace(os.sep, "/")] = _sha256(cand)
                break
        models.append({"name": name, "target": e.target, "backend": e.backend, "artefact": e.artefact,
                       "files_sha256": files})
    return [{"kind": "model-adm1 hybrid registry (models/*.yaml)", "use": list(hybrid_cfg.get("use") or []),
             "rate_overrides": summary.get("rate_overrides", []),
             "inhibition_overrides": summary.get("inhibition_overrides", []),
             "residual_correction": summary.get("residual_correction", []), "models": models}]


def main(argv=None):
    ap = argparse.ArgumentParser(description="ADM1 simulation + STAMM run package (stamm-model-core)")
    ap.add_argument("--scenario", help="scenario key of configs/scenarios.yaml (default: default_scenario)")
    ap.add_argument("--list-scenarios", action="store_true")
    ap.add_argument("--list-models", action="store_true", help="hybrid models registered in models/*.yaml")
    ap.add_argument("--list-hooks", action="store_true", help="what a hybrid model can plug into")
    ap.add_argument("--no-package", action="store_true", help="do not write the run package")
    ap.add_argument("--runs-dir", help="folder of the run packages (default: output.runs_dir)")
    ap.add_argument("--no-results", action="store_true", help="do not write results/dynamic_out.csv")
    ap.add_argument("--no-plots", action="store_true", help="do not write results/figures")
    ap.add_argument("--influent-hold", choices=["start", "end", "legacy"],
                    help="influent row held over [t_(i-1), t_i] (default: simulation.yaml influent_hold)")
    ap.add_argument("--t-end", type=float, help="shorter horizon [d] for a quick check (do not register such a "
                                                "package under the scenario's name: other values, same name)")
    a = ap.parse_args(argv)
    if a.list_hooks:
        print(format_hooks())
        return 0
    if a.list_models:
        print(format_registry(load_registry()))
        return 0
    try:
        cfg = load_config(ROOT)
        if a.list_scenarios:
            for k, s in cfg.scenarios.items():
                print(f"{k:24s} {' '.join(str(s.get('description', '')).split())[:90]}")
            return 0
        key = a.scenario or cfg.default_scenario
        scen = cfg.scenario(key)
        sim = cfg.simulation
        solver_cfg, time_cfg, output_cfg = sim.get("solver", {}), sim.get("time", {}), sim.get("output", {})
        method = str(solver_cfg.get("method", "BDF"))
        if method.upper() in EXPLICIT_SOLVERS and not bool(solver_cfg.get("allow_explicit", False)):
            raise SystemExit(f"solver.method = '{method}' is an explicit integrator; this ADM1 formulation is stiff and "
                             "explicit methods give wrong results (negative concentrations). Use 'BDF' (default), "
                             "'Radau' or 'LSODA', or set solver.allow_explicit: true in simulation.yaml.")

        # the parameters of the scenario (parameters.yaml + overrides), before the model recomputes any
        params, _ = apply_overrides(cfg.parameters, cfg.param_meta, scen.get("parameter_overrides"))
        init = InitialState(STATES_FILE, SCENARIOS_FILE, scenario_key=key, param=ADM1Parameters.from_values(params))
        x0 = init.get_dict()
        with open(STATES_FILE, encoding="utf-8") as fh:
            import yaml
            ph = ((yaml.safe_load(fh) or {}).get(init._get_active_key()) or {}).get("pH")
        if isinstance(ph, dict) and "value" in ph:
            x0["pH"] = float(ph["value"])               # declared pH (S_H_ion and S_anion follow from it)
        influent = Influent(INFLUENT_FILE, SCENARIOS_FILE, scenario_key=key, param=ADM1Parameters.from_values(params))
        hybrid_cfg = scen.get("adm1_hybrid") or {}
        registry = load_registry()
        hold = a.influent_hold or str(sim.get("influent_hold", "start"))
        model = ADM1(influent, hybrid_cfg, registry, piecewise=bool(solver_cfg.get("piecewise", True)),
                     influent_hold=hold)

        # horizon and output times (one row per influent row when time.dt_out is null)
        t_data = model.time_data
        t_start = _float(time_cfg.get("t_start", 0), 0)
        t_end_cfg = scen.get("t_end", time_cfg.get("t_end"))
        if t_end_cfg is not None:
            t_end = _float(t_end_cfg, 365.0)
        elif influent.mode == "dynamic":
            t_end = float(t_data[-1])
        else:
            t_end = _float(time_cfg.get("t_end_default_constant", 365.0), 365.0)
        if a.t_end is not None:
            t_end = float(a.t_end)
        if time_cfg.get("dt_out") is None and influent.mode == "dynamic":
            t_out = t_data[(t_data >= t_start) & (t_data <= t_end)].astype(float)
            if t_out[-1] < t_end:
                t_out = np.append(t_out, t_end)
            dt_out = float(np.median(np.diff(t_out))) if len(t_out) > 1 else float(t_end - t_start)
        else:
            dt_out = _float(time_cfg.get("dt_out"), 1.0)
            t_out = output_times(t_start, t_end, dt_out)

        solver = {"method": method, "rtol": solver_cfg.get("rtol", 1e-5), "atol": solver_cfg.get("atol", 1e-7),
                  "max_step": solver_cfg.get("max_step", 0.5)}
        if model.piecewise and len(t_data) > 1:
            # never let one solver step straddle several influent rows
            solver["max_step"] = min(float(solver["max_step"]), float(np.min(np.diff(t_data))))
        inputs_rec, extra_files = influent_record(influent, influent.mode_key)
        if influent.mode == "dynamic":
            inputs_rec["hold"] = hold      # which row is held over each interval: part of the scenario

        print(f"ADM1 {key}: influent {influent.mode_key} ({influent.mode}), {t_start:g} -> {t_end:g} d, "
              f"{len(t_out)} output rows, solver {method} rtol {solver['rtol']:g} atol {solver['atol']:g} "
              f"max_step {solver['max_step']:g}{', piecewise' if model.piecewise else ''}"
              f"{f', influent hold {hold}' if influent.mode == 'dynamic' else ''}", flush=True)
        out = run_scenario(model, ROOT, key, cfg=cfg, initial_state=x0, scenario_name=key, t_start=t_start, t_end=t_end,
                           t_out=t_out, dt_out=dt_out, solver=solver, inputs_record=inputs_rec, extra_files=extra_files,
                           hybrid_record=hybrid_record(model.hybrid_summary or {}, hybrid_cfg, registry),
                           config_files={"influent.yaml": INFLUENT_FILE, "initial_states.yaml": STATES_FILE},
                           package=not a.no_package and bool((output_cfg.get("run_package") or {}).get("enabled", True)),
                           runs_dir=a.runs_dir, write_results=not a.no_results and bool(output_cfg.get("save_results", True)))
    except StammModelError as exc:
        print(f"ERROR: {exc}")
        return 1
    if model.hybrid_summary and model.hybrid_summary.get("enabled"):
        s = model.hybrid_summary
        print("  hybrid: " + "; ".join(f"{k} {', '.join(s[k])}" for k in
                                       ("rate_overrides", "inhibition_overrides", "residual_correction") if s.get(k)))
    res = out.result
    c = res.columns
    print(f"{out.scenario_name}: {res.n_rows} rows, {c['time'][0]:g}-{c['time'][-1]:g} d, {res.duration_s:.1f} s, "
          f"{res.solver.get('segments')} solver segments; final pH = {c['pH'][-1]:.3f}, "
          f"q_ch4 = {c['q_ch4_norm_dry'][-1]:.4g} Nm3/d" + ("" if res.success else f"\n  RUN FAILED: {res.message}"))
    for w in out.warnings:
        print(f"  note: {w}")
    if out.results_csv:
        print(f"Results: {os.path.relpath(out.results_csv, ROOT)}")
    if out.zip_path:
        print(f"Run package: {os.path.relpath(out.zip_path, ROOT)}")
        print(out.validation.text())
    if not a.no_plots and bool(output_cfg.get("save_figures", True)) and res.n_rows >= 2 and hasattr(model, "table"):
        make_plots(model, output_cfg)
    if out.zip_path:
        return 0 if (out.validation.ok and res.success) else 2
    return 0 if res.success else 2


def make_plots(model, output_cfg):
    """results/figures: biogas, biomass, pH / alkalinity (a failed plot never fails the run)."""
    try:
        import matplotlib
        matplotlib.use("Agg" if not output_cfg.get("show_figures") else matplotlib.get_backend())
        from plots.plot_biogas import plot_biogas
        from plots.plot_biomass import plot_biomass
        from plots.plot_pH_alkalinity import plot_pH_alkalinity
    except Exception as exc:  # noqa: BLE001 - plots are optional
        print(f"Plots skipped: {exc}")
        return
    show = bool(output_cfg.get("show_figures", False))
    fig_dir = os.path.join(ROOT, output_cfg.get("output_dir", "results"), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    df, param = model.table, model.param
    for name, fn, kwargs in (("biogas", plot_biogas, {"param": param}), ("biomass", plot_biomass, {}),
                             ("pH_alkalinity", plot_pH_alkalinity, {"param": param})):
        try:
            fn(df=df, save_path=os.path.join(fig_dir, f"{name}.png"), show=show, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"{name} plot skipped: {exc}")


if __name__ == "__main__":
    sys.exit(main())

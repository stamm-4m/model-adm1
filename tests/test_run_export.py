"""
Tests of the STAMM run package (src/run_export.py, docs/run_package.md).

No ODE integration: a synthetic results table is exported with the real configuration objects
of the active scenario, so the test runs in a second and does not need scipy.

    python tests/test_run_export.py        (or: pytest tests/test_run_export.py)
"""
import io
import json
import os
import sys
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml

from initial_states import InitialState
from src.acid_base import compute_acid_base_equilibrium
from src.influent import Influent
from src.parameters import ADM1Parameters
from src.reactor import ADM1Reactor, FULL_STATE_NAMES
from src.run_export import collect_parameters, export_run_package


def _build_package(runs_dir):
    scenario_cfg = yaml.safe_load(open("configs/scenario.yaml", encoding="utf-8"))
    name = scenario_cfg["active_scenario"]
    block = scenario_cfg["scenarios"][name]

    param = ADM1Parameters()
    init_state = InitialState()
    declared = init_state.get_dict()
    y0 = np.asarray(init_state.get_vector(), dtype=float)
    influent = Influent()
    reactor = ADM1Reactor(param, constants=None)
    snapshot = dict(param.params)
    eq = compute_acid_base_equilibrium(reactor.unpack_state(y0), param)
    y0_eff = reactor._apply_acid_base_to_state_vector(y0, eq)

    # synthetic results table with the same column layout as main.py
    df = pd.DataFrame([y0_eff, y0_eff], columns=FULL_STATE_NAMES)
    df.insert(0, "time", [0.0, 1.0])
    out_vars = yaml.safe_load(open("configs/output_variables.yaml", encoding="utf-8"))
    for col in out_vars:
        if col not in df.columns:
            df[col] = 0.0

    run_dir = export_run_package(
        df=df, scenario_name=name, scenario_block=block, param_snapshot=snapshot,
        declared_initial=declared, effective_initial=y0_eff, state_names=FULL_STATE_NAMES,
        states_key=init_state._get_active_key(), influent=influent,
        hybrid_cfg=block.get("hybrid", {}), hybrid_summary={"enabled": False},
        solver={"method": "BDF"}, time_info={"t_start": 0.0, "t_end": 1.0, "dt_out": 1.0, "units": "d"},
        status={"success": True, "message": "synthetic", "warnings": []},
        started_at=time.time(), package_cfg={"runs_dir": runs_dir, "zip": True},
    )
    return run_dir, df, block, influent


def test_run_package():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir, df, block, influent = _build_package(tmp)
        run_id = os.path.basename(run_dir)

        # 1. files
        files = sorted(os.listdir(run_dir))
        expected = ["dynamic_out.csv", "run.json"] + (["influent.csv"] if influent.mode == "dynamic" else [])
        assert files == sorted(expected), files
        with zipfile.ZipFile(os.path.join(tmp, run_id + ".zip")) as zf:
            assert sorted(zf.namelist()) == sorted(expected)          # flat archive

        # 2. the CSV is exactly what main.py writes (df.to_csv(index=False))
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        with open(os.path.join(run_dir, "dynamic_out.csv"), encoding="utf-8", newline="") as f:
            assert f.read().replace("\r\n", "\n") == buf.getvalue().replace("\r\n", "\n")

        # 3. strict JSON (no NaN / Infinity) with the agreed sections
        with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
            r = json.loads(f.read(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        assert list(r) == ["schema_version", "model", "calibration", "run", "outputs", "reference"]
        assert r["run"]["run_id"] == run_id

        # 4. every CSV column is described, none unknown
        variables = r["outputs"]["variables"]
        assert [v["name"] for v in variables] == list(df.columns)
        assert all(v["kind"] in ("time", "state", "algebraic", "derived") for v in variables)
        assert all(v["units"] for v in variables)

        # 5. parameters: every scenario override is flagged, with its default value
        params = {p["name"]: p for p in r["calibration"]["parameters"]}
        for name in (block.get("parameter_overrides") or {}):
            assert params[name]["source"] == "scenario", name
            assert "default_value" in params[name]
        # K_w is recomputed from T_op by ADM1Reactor -> "computed"
        assert params["K_w"]["source"] in ("computed", "default")
        assert all(p["source"] in ("default", "scenario", "computed") for p in params.values())

        # 6. state variables: 38 states, effective vs declared initial values
        states = [s for s in r["calibration"]["state_variables"] if s["kind"] != "declared_only"]
        assert [s["symbol"] for s in states] == FULL_STATE_NAMES
        assert all(s["initial"] is not None for s in states)

        # 7. influent
        inf = r["calibration"]["influent"]
        assert inf["mode"] == influent.mode
        if influent.mode == "dynamic":
            assert inf["file"] == "influent.csv" and len(inf["sha256"]) == 64
        else:
            assert inf["values"]


def test_parameter_sources(tmp_path=None):
    tmp = tempfile.mkdtemp()
    base = os.path.join(tmp, "p.yaml")
    with open(base, "w", encoding="utf-8") as f:
        yaml.safe_dump({"g": {"a": {"value": 1.0, "units": "-"}, "b": {"value": 2.0}, "c": {"value": 3.0}}}, f)
    rows = collect_parameters(
        base,
        {"parameter_overrides": {"a": {"value": 5.0, "description": "set by hand"}, "z": {"value": 9.0}}},
        {"a": 5.0, "b": 2.0, "c": 3.5, "z": 9.0},
    )
    by = {r["name"]: r for r in rows}
    assert by["a"]["source"] == "scenario" and by["a"]["default_value"] == 1.0
    assert by["a"]["override_note"] == "set by hand"
    assert by["b"]["source"] == "default" and "default_value" not in by["b"]
    assert by["c"]["source"] == "computed" and by["c"]["default_value"] == 3.0
    assert by["z"]["group"] == "scenario_only" and by["z"]["source"] == "scenario"


if __name__ == "__main__":
    test_parameter_sources()
    test_run_package()
    print("test_run_export: all checks passed ✅")

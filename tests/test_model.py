"""
Tests of model-adm1 on stamm-model-core (backlog 15, step 3, 2026-11-06): short runs of a dynamic
and a constant-influent scenario written as STAMM run packages, checked with the registry's rules.

    python tests/test_model.py
"""
import json
import os
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import main as adm1_main  # noqa: E402
from stamm_model import apply_overrides, load_config, validate_package  # noqa: E402
from stamm_model.values import StateVector, Values  # noqa: E402

from initial_states import InitialState  # noqa: E402
from model import ADM1  # noqa: E402
from src.influent import Influent  # noqa: E402
from src.parameters import ADM1Parameters  # noqa: E402
from src.reactor import DYNAMIC_STATE_NAMES  # noqa: E402


def _run(key, t_end):
    runs = tempfile.mkdtemp(prefix="adm1_test_")
    rc = adm1_main.main(["--scenario", key, "--t-end", str(t_end), "--runs-dir", runs, "--no-results", "--no-plots"])
    assert rc == 0, f"{key}: exit {rc}"
    zips = [os.path.join(runs, f) for f in os.listdir(runs) if f.endswith(".zip")]
    assert len(zips) == 1, zips
    rep = validate_package(zips[0])
    assert rep.ok, rep.text()
    with zipfile.ZipFile(zips[0]) as zf:
        return zf, json.loads(zf.read("run.json")), zf.read("dynamic_out.csv").decode(), zf.namelist()


def test_dynamic_short_run():
    _, doc, csv_text, names = _run("BSM2_ringtest", 3)
    header = csv_text.splitlines()[0].split(",")
    with open("configs/outputs.yaml", encoding="utf-8") as fh:
        assert header == [v["name"] for v in yaml.safe_load(fh)["variables"]]
    assert len(csv_text.splitlines()) - 1 == 4                       # t = 0, 1, 2, 3 (daily influent rows)
    params = {p["name"]: p for p in doc["calibration"]["parameters"]}
    assert all(params[k]["source"] == "computed" for k in ("K_w", "K_a_co2", "K_a_IN"))
    assert params["T_op"]["value"] == 308.15
    inp = doc["calibration"]["inputs"]
    assert inp["mode_key"] == "dynamic_Q" and "influent.csv" in names
    assert inp["files"]["influent.csv"] == inp["sha256"]
    assert doc["run"]["solver"]["segments"] == 3


def test_constant_short_run():
    _, doc, csv_text, _ = _run("BSM2_constant", 2)
    assert len(csv_text.splitlines()) - 1 == 3                       # t = 0, 1, 2
    assert doc["run"]["solver"]["segments"] == 1                      # constant feed: one solver call
    inp = doc["calibration"]["inputs"]
    assert inp["mode"] == "constant" and inp["values"] and "files" not in inp


def test_rhs_is_the_reactor():
    key = "BSM2_dynamic"
    cfg = load_config(ROOT)
    params, _ = apply_overrides(cfg.parameters, cfg.param_meta, cfg.scenario(key).get("parameter_overrides"))
    x0 = InitialState(scenario_key=key, param=ADM1Parameters.from_values(params)).get_dict()
    model = ADM1(Influent(scenario_key=key, param=ADM1Parameters.from_values(params)), progress_s=0)
    y0 = model.initial_state(x0, Values(params, "parameter"))
    y = np.array([y0[s] for s in DYNAMIC_STATE_NAMES])
    d = model.rhs(0.0, StateVector(y, {s: i for i, s in enumerate(DYNAMIC_STATE_NAMES)}), None, None)
    assert np.shape(d) == (len(DYNAMIC_STATE_NAMES),) and np.all(np.isfinite(d))
    assert set(model.derived_parameters(None)) == {"K_w", "K_a_co2", "K_a_IN"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} OK")
    print("ALL TESTS PASSED")

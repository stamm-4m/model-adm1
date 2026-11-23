# Run package for the STAMM model registry

Every run of `python main.py` writes, **in addition to** `results/dynamic_out.csv` and the figures,
a self-describing package that can be uploaded to the STAMM model registry (Dynamic Models >
Register simulations):

```
runs/<run_id>/
├── dynamic_out.csv    simulation trajectories (same content as results/dynamic_out.csv)
├── run.json           model, calibration (= scenario), run, outputs, reference
└── influent.csv       copy of the influent CSV (dynamic influent only)
runs/<run_id>.zip      the same files, flat archive, ready to upload
```

Since 2026-11-06 the package is written by **stamm-model-core** (schema 1.0 of the STAMM registry),
the package model-adm1 runs on; `src/run_export.py`, the previous writer, was removed. After writing
it, the core checks it with the registry's own rules and prints the result (`VALID run package` or
the list of problems).

- `run_id` = `<YYYYMMDDTHHMMSS>_<scenario>_<6-char fingerprint>`; the fingerprint is computed from
  the scenario name, parameters, initial set, influent definition and hybrid models, so two runs of
  the same scenario with the same values share it. Runs are never overwritten.
- `results/dynamic_out.csv` is written before the package: if the package cannot be written, the
  run reports the error (`ERROR: ...`, exit code 1) and the results file is still there.
- `--no-package` or `output.run_package.enabled: false` turns the package off; `--runs-dir` writes
  it elsewhere (default `output.runs_dir`).

## Configuration

`configs/simulation.yaml`:

```yaml
output:
  results_dir: "results/"
  runs_dir: "runs/"         # one sub-folder per run + its zip
  zip: true
  code_files: ["main.py", "model.py", "initial_states.py", "src/*.py"]   # hashed in run.json
  run_package:
    enabled: true
```

Three configuration files describe the model to the core:

| File | Content |
|---|---|
| `configs/model_card.yaml` | identity of the model: name, implementation, **version** (bump it at each release that changes results), authors, references, license |
| `configs/outputs.yaml` | **every** CSV column, in CSV order: `name`, `kind`, `units`, `label`, `description` |
| `configs/states.yaml` | the 29 integrated states, in solver order, with units |

A column the run produces but `outputs.yaml` does not list is left out of the CSV, with a warning in
`run.json`; a column listed but not produced is an error.

## How `run.json` maps to the registry

| Level | Meaning | `run.json` section |
|---|---|---|
| Family | the model code (ADM1, version X) | `model` |
| Scenario | parameters, initial set, inputs, hybrid models | `calibration` |
| Simulation | one execution of a scenario | `run`, `outputs`, `reference` |

The registry identifies a scenario by its name and a fingerprint of `calibration` (scenario,
parameters, state variables, inputs, hybrid): a scenario with other values must get another name.

## `run.json` field by field

### `model` — family

| Field | Description |
|---|---|
| `name`, `implementation`, `version`, `model_kind`, `description`, `time_units`, `authors`, `references`, `repository`, `license` | from `configs/model_card.yaml` |
| `code_ref.git_commit`, `git_branch`, `uncommitted_changes` | read-only git queries (null without git) |
| `code_ref.source_sha256`, `source_files` | hash of the files of `output.code_files`: identifies the exact code even without git |

### `calibration` — scenario

| Field | Description |
|---|---|
| `scenario` | `name` (the key of `scenarios.yaml`), `description`, `kind` (`continuous`), `t_end` |
| `parameters[]` | **all** parameters used (see below) |
| `state_variables[]` | the initial set as declared: `symbol`, `name`, `kind`, `units`, `initial` (with `pH` when the set gives one; S_H_ion / S_anion are then derived from it) |
| `inputs` | the influent: `kind` (`influent`), `mode_key`, `mode`, `description`, `hold` (`influent_hold`); dynamic: `influent_file` (`influent.csv`), `source_path`, `sha256`, `time_column`, `flow_column`, `time_units`, `n_rows`, `t_first`, `t_last`, and `files` = sha256 of the copy in the package; constant: `values[]` = `{name, value, units}` |
| `hybrid` | null, or the models of `models/*.yaml` plugged in: `use`, the hooks wired (`rate_overrides`, `inhibition_overrides`, `residual_correction`) and, per model, `target`, `backend`, `artefact` and the sha256 of its files |
| `config_files_sha256` | hash of every configuration file read (the stamm-model-core files plus `influent.yaml` and `initial_states.yaml`) |

**`parameters[]`**

| Field | Description |
|---|---|
| `name`, `group` | `group` is the YAML section of `parameters.yaml` |
| `value` | value used by the run |
| `units`, `description`, `reference` | from `parameters.yaml` (`reference` = its `source`) |
| `source` | `"default"`: `parameters.yaml`<br>`"scenario"`: overridden by the scenario's `parameter_overrides` (T_op included)<br>`"computed"`: recomputed by the model before the run: `K_w`, `K_a_co2`, `K_a_IN` from `T_op` |
| `default_value`, `override_note` | for `scenario`: the `parameters.yaml` value and the description written next to the override; for `computed`: the `parameters.yaml` value |

A dynamic `Q(t)` (influent `flow_column`) changes `q_ad` during the run; the recorded `q_ad` is the
value before the run, and the flow actually applied is in `influent.csv`.

### `run` — simulation

| Field | Description |
|---|---|
| `run_id`, `created_at`, `user`, `duration_s` | — |
| `solver` | `method`, `rtol`, `atol`, `nfev` (right-hand-side evaluations), `segments` (solver pieces: one per influent interval with a dynamic influent) |
| `time` | `t_start`, `t_end`, `dt_out`, `units`, `n_output_points`, `mode` |
| `status` | `success`, `message`, `warnings[]`: model events, a negative concentration ("results NOT valid"), undocumented columns |
| `environment` | Python, platform, numpy, scipy, stamm-model-core versions |

### `outputs`

| Field | Description |
|---|---|
| `file`, `sha256`, `n_rows`, `time_column` | — |
| `variables[]` | the entries of `configs/outputs.yaml`, in CSV order |

Values of `kind`: `time`; `state` (integrated); `algebraic` (acid–base species from the charge
balance); `derived` (pH, charge residual, COD diagnostics, biogas flows). STAMM decides at ingestion
which columns it uses, based on `kind`.

### `reference` — optional

Measured digester data are not copied into the package: they belong to STAMM's experiment data.
No model-adm1 scenario sets a reference today (`reference: null`). stamm-model-core can compute one
from a CSV of measurements given in the scenario (`reference:` block, see the stamm-model-core guide).

## Tests

```bash
python tests/test_model.py      # two short runs (dynamic, constant) packaged and checked
python tools/compare_port.py    # the port to stamm-model-core against the previous code
```

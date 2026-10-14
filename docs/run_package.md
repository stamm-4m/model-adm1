# Run package for the STAMM model registry

Every run of `python main.py` writes, **in addition to** the usual `results/` files, a
self-describing package that can be uploaded to the STAMM model registry:

```
runs/<run_id>/
├── dynamic_out.csv    simulation trajectories — byte-identical to results/dynamic_out.csv
├── run.json           everything else: model, calibration, run, outputs, reference
└── influent.csv       copy of the influent CSV (dynamic influent only)
runs/<run_id>.zip      the same files, flat archive, ready to upload
```

Guiding principle: **add without breaking**. The CSV (columns, order, file name) is unchanged, and
`results/dynamic_out.csv` and the figures are still written as before. All new information lives in
`run.json`.

- `run_id` = `<YYYYMMDDTHHMMSS>_<scenario>_<6-char fingerprint>`, so runs are never overwritten.
- The package is written by `src/run_export.py`, called at the end of step 10 of `main.py`.
- If anything goes wrong while writing the package, the simulation is **not** affected. The console
  prints `Run package skipped: <reason>`.

## Configuration

`configs/simulation.yaml`:

```yaml
output:
  run_package:
    enabled: true        # false = no package (results/ only, as before)
    runs_dir: "runs/"    # one sub-folder per run
    zip: true            # also write runs/<run_id>.zip
```

Two configuration files feed the package:

| File | Content |
|---|---|
| `configs/model_card.yaml` | identity of the model family: name, implementation, **version** (bump it manually at each release), authors, references |
| `configs/output_variables.yaml` | dictionary of **every** CSV column: `kind`, `units`, `name`, `description` |

When a new column is added to the CSV, add it to `output_variables.yaml`. Otherwise, `run.json`
reports it under `run.status.warnings`, and `tests/test_run_export.py` fails.

## How `run.json` maps to the registry

STAMM stays generic: dynamic models in general, not anaerobic digestion specifically. Each section of
`run.json` maps to one registry level.

| Level | Meaning | `run.json` section |
|---|---|---|
| Family | the model code (ADM1, version X) | `model` |
| Instance | one calibration: parameters, initial states, inputs, switches | `calibration` |
| Simulation | one execution of an instance | `run`, `outputs`, `reference` |

Keys reuse the vocabulary of STAMM's `dynamic_model.information` JSON (`state_variables`,
`parameters`, …).

## `run.json` field by field

### `schema_version`

`"1.0"`

### `model` — family

| Field | Description |
|---|---|
| `name`, `implementation`, `version`, `model_kind`, `description`, `time_units`, `authors`, `references`, `repository`, `license` | from `configs/model_card.yaml` |
| `code_ref.git_commit`, `code_ref.git_branch` | read-only `git rev-parse` (null without git) |
| `code_ref.uncommitted_changes` | `true` if `main.py`, `initial_states.py` or `src/*.py` differ from the commit (read-only `git diff --quiet`) |
| `code_ref.source_sha256` | hash of `main.py`, `initial_states.py` and `src/*.py`; always present, identifies the exact code even without git |
| `code_ref.source_files` | the files included in that hash |

### `calibration` — instance

| Field | Description |
|---|---|
| `scenario` | `name`, `description`, `initial_states` (key), `influent_mode` (key), `t_end` |
| `parameters[]` | **all** resolved parameters (see below) |
| `state_variables[]` | the 38 states with their initial values (see below) |
| `influent` | `mode_key`, `mode` (`dynamic` / `constant`), `description`; for dynamic: `file` (`influent.csv`), `source_path`, `sha256`, `time_column`, `flow_column`, `n_rows`, `t_first`, `t_last`; for constant: `values[]` = `{name, value, units}` |
| `hybrid` | scenario `hybrid` block + the hooks actually wired (`rate_overrides`, `inhibition_overrides`, `residual_correction`) |
| `config_files_sha256` | hash of the five configuration YAML files used |

**`parameters[]`**

| Field | Description |
|---|---|
| `name`, `group` | `group` is the YAML section (`stoichiometric_parameters`, `biochemical_parameters`, `disintegration`, `gas_flow`, …) |
| `value` | value actually used, taken **before** integration |
| `units`, `description` | from `adm1_parameters.yaml` |
| `source` | `"default"`: base YAML (literature)<br>`"scenario"`: overridden by the active scenario (`parameter_overrides` or `T_op`)<br>`"computed"`: changed by the code before integration (see below) |
| `default_value` | base YAML value, present when `source` is `scenario` or `computed` |
| `override_note` | the `description` written next to the override in `scenario.yaml` |

Filtering on `source == "scenario"` shows exactly what the modeller changed. Some parameters are
`"computed"`, not `"default"`: `K_w`, `K_a_co2` and `K_a_IN` are recomputed from `T_op` in
`ADM1Reactor.__init__`. The value used is therefore not exactly the YAML value.

Overrides of keys that do not exist in the base YAML appear with `group: "scenario_only"`. The
`T_ad` alias appears with `group: "derived"`.

The snapshot is taken right after the reactor is built, so a dynamic `Q(t)` (which changes `q_ad`
during the run) does not affect the recorded value.

**`state_variables[]`**

| Field | Description |
|---|---|
| `symbol`, `name`, `kind`, `units`, `description` | from `output_variables.yaml` (`units` = CSV units) |
| `initial` | **effective** initial value used by the solver |
| `declared_initial` | value declared in `initial_states.yaml` (with `S_H_ion` / `S_anion` derived from `pH` when the set gives one) |
| `declared_units` | only when the YAML unit really differs from the CSV unit |

`initial` and `declared_initial` can differ because `main.py` re-projects the initial state onto the
acid–base equilibrium before integration. For `S_H_ion`, `S_hco3_ion`, `S_nh3`, the VFA ions, etc.,
the run starts from a value that differs from the YAML. `declared_units` makes a real unit difference
explicit. For example, the VFA ions are declared in kgCOD.m⁻³ in `initial_states.yaml` but are kmol.m⁻³
(= M) in the CSV. `M` = `kmol.m^-3`, and `kmol C.m^-3` = `kmol.m^-3`, are not reported as differences.

### `run` — simulation

| Field | Description |
|---|---|
| `run_id` | same as the folder name |
| `created_at` | ISO 8601, local time with UTC offset |
| `user` | OS user name |
| `duration_s` | wall-clock duration of the whole run |
| `solver` | `method`, `rtol`, `atol`, `max_step`, `piecewise` |
| `time` | `t_start`, `t_end`, `dt_out`, `units`, `n_output_points` |
| `status` | `success`, `message`, `warnings[]` (negative-concentration guard, undocumented CSV columns) |
| `environment` | Python, platform, numpy, scipy, pandas, PyYAML, matplotlib versions |

### `outputs`

| Field | Description |
|---|---|
| `file` | `"dynamic_out.csv"` |
| `sha256`, `n_rows`, `n_columns`, `time_column` | — |
| `variables[]` | one entry per CSV column, in CSV order: `name`, `label`, `units`, `kind`, `description` |

Possible values of `kind`:

- `time`: the time axis;
- `state`: integrated ODE state;
- `algebraic`: acid–base species recomputed from the charge balance;
- `derived`: pH, charge residual, COD diagnostics, biogas flows.

The model does not filter columns. STAMM decides at ingestion which columns it uses, based on
`kind`.

### `reference` — optional

Measured digester data are **not** copied into the package: they belong to STAMM's experiment data.
To link a run to an experiment, add a `reference` block to the scenario in `configs/scenario.yaml`.
It is copied as-is into `run.json`. Otherwise, `reference` is `null`.

```yaml
  my_scenario:
    ...
    reference:
      experiment: "<experiment name or STAMM id>"
      description: "what the run is compared against"
      metrics:                       # optional, filled by whoever computes the fit
        - {variable: q_ch4_norm_dry, rmse: 0.0, r2: 0.0}
```

## Test

```bash
python tests/test_run_export.py
```

The test runs without ODE integration (synthetic table, real configuration objects, about 1 s). It checks:

- the files and the flat zip;
- that the package CSV equals `df.to_csv`;
- strict JSON (no NaN);
- that every CSV column is documented;
- the parameter `source` flags;
- the 38 states;
- the influent description.

## Next steps (STAMM side, out of scope here)

- Database:
  - extend `dynamic_model` as the family level;
  - add an instance table;
  - replace the `simulations` table with a run table whose trajectories are stored as files or in long format.
- Upload endpoint that ingests `runs/<run_id>.zip`.
- Generic JSON Schema for `run.json`, reusable by other dynamic models.

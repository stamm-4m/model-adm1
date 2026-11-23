# 2026-11-06 — model-adm1 on stamm-model-core (STAMM backlog 15, step 3)

## What changed
model-adm1 now runs on `stamm-model-core` 0.2.1 (forge.inrae.fr/stamm/models/stamm-model-core).
The science is unchanged: `src/reactor.py`, `src/acid_base.py`, `src/hybrid.py`, `src/registry.py`,
`models/*.yaml`, `examples/` (comments only: the new file names).

- `model.py` (new): `ADM1(OdeModel)` around `ADM1Reactor`. 29 integrated states (the 9 acid-base
  species stay algebraic, solved inside every RHS call). Hooks: `initial_state` (initial set,
  hybrid hooks wired, acid-base re-projection), `derived_parameters` (K_w, K_a_co2, K_a_IN
  recomputed from T_op, reported with source "computed"), `breakpoints` (the solver restarts at
  every influent row, "piecewise"), `rhs` (influent read by the model, zero-order hold, dynamic
  Q(t)), `post_columns` (acid-base species, pH, COD and biogas flows in the same order as before:
  the acid-base Newton starts from the previous solution), `result_warnings` (a negative
  concentration flags the run as numerically invalid).
- `main.py` (rewritten): `--scenario` instead of editing `active_scenario`; `--list-scenarios`,
  `--list-models`, `--list-hooks`, `--no-package`, `--runs-dir`, `--no-results`, `--no-plots`,
  `--t-end` (quick checks only). Output times, solver settings (max_step bounded by the influent
  step) and horizon computed as before. Figures as before (`results/figures`).
- Run package written by stamm-model-core: `dynamic_out.csv` (same columns and values), `influent.csv`
  (dynamic influent), `run.json` with the influent definition (`calibration.inputs`, sha256), the
  hybrid models used with the sha256 of their files (`calibration.hybrid`) and the sha256 of
  `influent.yaml` and `initial_states.yaml` too.
- Loaders: `ADM1Parameters`, `InitialState`, `Influent` take `scenario_key=` and `param=`
  (`ADM1Parameters.from_values`); defaults point to the new file names.
- Configuration in the core layout (values checked one by one by the migration script):
  `parameters.yaml` (was `adm1_parameters.yaml`; + `source` per group, from the references of the
  file — to refine per parameter), `scenarios.yaml` (was `scenario.yaml`; T_op moved into
  `parameter_overrides`, `hybrid` renamed `adm1_hybrid`, `default_scenario: BSM2_ringtest`),
  `outputs.yaml` (was `output_variables.yaml`, CSV column order), `states.yaml` (new, 29 states),
  `simulation.yaml` + `mode`, `dt_out`, `t_start`, `include_initial`, `output.results_dir /
  runs_dir / zip / code_files`.
- Removed (copies in `STAMM/backups/model_adm1_original_20261106/`): `configs/adm1_parameters.yaml`,
  `configs/scenario.yaml`, `configs/output_variables.yaml`, `src/run_export.py`,
  `tests/test_run_export.py` (replaced by `tests/test_model.py`).
- `.gitignore`: `runs/`, `results/dynamic_out*.csv`, `results/figures/`; Tatiana's lab-digester data
  (`data/influent/lab_5L_2023/`, `tests/data/tatiana_rm_sim_result.*`, `tests/compare_tatiana_rm.py`)
  not published until she agrees (Camilo). Her two scenarios stay in `scenarios.yaml`.
- `requirements.txt`: + `stamm-model-core>=0.2.1`. License unchanged (Apache-2.0; README note on
  the AGPL-3.0 of the whole with stamm-model-core).

Additions to stamm-model-core for this port (0.2.1, `tests/test_forcing.py`): a piece of a
continuous run is always integrated up to its end (bug fix: with a cut between two output times
the next piece started from the last output row — 8 % error in the new test), `breakpoints()`,
explicit output times `t_out`, `solver` overrides, `derived_parameters()`, `extra_files`,
`hybrid_record`, `config_files`, `result_warnings()`.

## Checked (Camilo's PC, 2026-11-06)
`python tools/compare_port.py` (previous code = the untouched copy in backups, run side by side):
BSM2_ringtest, BSM2_dynamic, BSM2_steady_state, BSM2_constant, thermophilic, batch_validation,
pig_slurry_test, no_xc_demo, hybrid_demo, hybrid_lr_demo, hybrid_lr_spec_demo:
- `dynamic_out.csv` identical in all 11 (same header, same values; only the line endings differ:
  the previous code wrote it with pandas, CRLF on Windows), `influent.csv` identical;
- parameter values identical (the previous package also listed T_ad, an alias of T_op no equation
  reads);
- every new package accepted by the registry's own parser; same warnings.
BSM2_ringtest_15min (run separately, `--only BSM2_ringtest_15min`): 26 881 rows identical, influent.csv
identical, same parameters, accepted (previous 1280 s, new 1347 s); `tools/compare_port_report.txt`
holds this last run.
Tests: `tests/test_model.py` 3/3, stamm-model-core 44/44.

## Found, not changed
- `hybrid_lr_demo` and `hybrid_lr_spec_demo` (linear regressions fitted on synthetic data replacing
  Rho_2) drive X_ch negative (-7.8 at day 98, -0.13 at day 147): results NOT valid, in the previous
  code as well. Now written in run.json -> run.status.warnings. Backlog 62.
- At the last RHS evaluations of an influent interval (t = t_i exactly) the influent lookup returns
  the next row (searchsorted side "right"), whereas the comment of the previous main.py says row i
  is active on (t_(i-1), t_i]. Small effect; the BSM2 validation passed with it. Kept for identical
  results here; corrected the same day (backlog 63): docs/CHANGES_2026-11-06_influent_hold_63.md.
- `docs/` (user manual, configuration, run package, ...) still describe the previous file names and
  `active_scenario`; a note at the top of each points here. Backlog 64.

# 2026-11-07 — Documentation on the stamm-model-core layout (STAMM backlog 64) and hybrid demos (62)

## 64: docs/ on the new layout
The user manual, configuration, class diagram, architecture, hybrid guide, validation guide, the
documentation index and the READMEs of examples/, models/ and data/ now use `python main.py
--scenario <key>`, `configs/scenarios.yaml` (T_op in `parameter_overrides`, `adm1_hybrid`),
`parameters.yaml`, `outputs.yaml`, `states.yaml`, `model_card.yaml` and `influent_hold`; the
provisional notes of 2026-11-06 are removed. `docs/run_package.md` is rewritten for the package
written by stamm-model-core (computed parameters, influent with its hold and sha256, hybrid models
with the sha256 of their files, check with the registry's rules). The dated documents (CHANGES_*,
validation_report_2026-09-08.md) keep the names of their time.

## 62: hybrid demos
The models of examples/ and models/ are fictitious: never trained or tested on real data (Camilo,
2026-11-07). Nothing to fix in them; they are now labelled: a note in examples/README.md,
models/README.md, docs/hybrid.md and the user manual, and "ILLUSTRATIVE ONLY: fictitious plug-in
models ..." in the description of the three hybrid scenarios of configs/scenarios.yaml (written in
run.json and the registry with the scenario). The two linear-regression demos also say "Results NOT
valid: the synthetic regression drives X_ch negative" (day 98 / day 147); their runs carry the same
warning in run.json -> run.status.warnings. Nothing else changes (no code, no values).

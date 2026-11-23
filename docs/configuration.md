# Configuration files — design rationale and per-file reference

**Authors**
- Margaux Bonal — <margaux.bonal@inrae.fr>
- David Camilo Corrales — <David-Camilo.Corrales-Munoz@inrae.fr>

**Date:** April 2026 (updated 2026-11-07: stamm-model-core layout)

This document explains how the ADM1 configuration is split across the
YAML files in [`../configs/`](../configs/), what each one is for,
what it should and shouldn't contain, and why it's organised this way.
Read this if you want to add a new scenario, calibrate parameters, or
extend the configuration.

---

## TL;DR — one file per question

The configuration follows a **separation of responsibilities**: each
file answers exactly one question about the simulation. Six files for
the model and its scenarios, plus three that describe the model to
stamm-model-core (the package that runs it and writes the STAMM run
package).

| Question | File | What lives here |
| --- | --- | --- |
| What are the reference ADM1 parameters? | [`configs/parameters.yaml`](../configs/parameters.yaml) | All intrinsic ADM1 constants — kinetic, stoichiometric, physico-chemical. **Independent of any specific simulation.** |
| What is the starting state of the digester? | [`configs/initial_states.yaml`](../configs/initial_states.yaml) | Named initial-state vectors (BSM2 reference, custom start-ups, ...). |
| What enters the digester? | [`configs/influent.yaml`](../configs/influent.yaml) | Influent definitions: dynamic CSV time series or constant values. |
| What do we want to test? | [`configs/scenarios.yaml`](../configs/scenarios.yaml) | Named scenarios (`python main.py --scenario <name>`, `default_scenario`) + per-scenario overrides. **Main entry point for the user.** |
| How are the equations solved? | [`configs/simulation.yaml`](../configs/simulation.yaml) | ODE solver settings, time horizon, output options. |
| Which parameters are adjusted? | [`configs/calibration.yaml`](../configs/calibration.yaml) | Calibration framework: free parameters, bounds, objective function (not read by any script yet). |
| Which variables does the solver integrate? | [`configs/states.yaml`](../configs/states.yaml) | The 29 integrated states in solver order, with units (stamm-model-core). |
| What is in the results CSV? | [`configs/outputs.yaml`](../configs/outputs.yaml) | Every column of `dynamic_out.csv`, in order: kind, units, label, description. |
| Which model is this? | [`configs/model_card.yaml`](../configs/model_card.yaml) | Name, version, authors, references, license: written in every run package. |

This split is what lets you:

- **Compare simulation cases** without touching the model parameters
  (only swap scenarios).
- **Calibrate parameters** without entangling them with operating
  conditions (free parameters live in `calibration.yaml`, not in
  `parameters.yaml`).
- **Switch operating regimes** (mesophilic ↔ thermophilic) without
  forking the parameter file (use `parameter_overrides:` in
  `scenarios.yaml`).
- **Reproduce results** because each file has a single authoritative
  source of truth.

---

## Why split, instead of one big file?

The original PyADM1 implementation kept everything in a single
configuration file. That's quick to set up but becomes limiting when:

- Comparing multiple simulation cases — every fork duplicates the whole
  parameter set, including the parts that didn't change.
- Modifying operating conditions on the fly — changing one number in a
  monolithic file makes the diff hard to read and audit.
- Calibrating parameters — calibration code wants a clear list of
  *which* parameters are free and *what their bounds are*, separate from
  the fixed model parameters.
- Toggling model components — turning a feature on or off requires
  hunting through the file rather than flipping a flag in a known place.

The six-file layout fixes all four. Each file is small, has one job, and
can be reviewed independently.

---

## Per-file reference

### [`configs/parameters.yaml`](../configs/parameters.yaml) — the model

The **central, stable parameter library** for ADM1. Values from
Rosen et al. (2006) BSM2 report and Batstone et al. (2002). Each entry has
`value`, `units`, `source` and `description` (stamm-model-core requires the
four); the sources are given per group for now.

Includes:
- General constants (e.g. `R`, `T_base`, `p_atm`)
- Stoichiometric parameters (`f_*`, `Y_*`, elemental coefficients `C_*`, `N_*`)
- Biochemical and kinetic parameters (`k_dis`, `k_hyd_*`, `k_m_*`, `K_S_*`, `k_dec_*`)
- Inhibition parameters (`K_I_*`, `pH_LL_*`, `pH_UL_*`)
- Physico-chemical parameters (`K_a_*`, `K_H_*`, `K_w`, `k_p`, `k_L_a`)
- Default reactor parameters (`V_liq`, `V_gas`, `q_ad`, `T_op`)

Must **not** contain simulation-dependent elements such as initial
states, influent profiles, numerical settings, scenarios, or calibration
parameters.

> Treat this file as a stable library. Per-scenario tweaks belong in
> `scenarios.yaml > parameter_overrides:`, not here. Values the code
> recomputes (K_w, K_a_co2, K_a_IN from T_op) are reported in run.json with
> source "computed".

### [`configs/initial_states.yaml`](../configs/initial_states.yaml) — where the digester starts

Named initial-state vectors. Each set defines the 38 ADM1 state
variables at `t = 0`:

- Dissolved states (`S_su`, `S_aa`, `S_fa`, `S_va`, `S_bu`, `S_pro`, `S_ac`, `S_h2`, `S_ch4`, `S_IC`, `S_IN`, `S_I`)
- Particulate states (`X_xc`, `X_ch`, `X_pr`, `X_li`, biomass `X_*`, `X_I`)
- Ionic states + pH (`S_cation`, `S_anion`, `S_H_ion`, `S_*_ion`, `S_co2`, `S_nh3`, `S_nh4_ion`)
- Gas-phase states (`S_gas_h2`, `S_gas_ch4`, `S_gas_co2`)

Multiple sets can coexist (e.g. `BSM2` reference, plus custom start-ups
or thermophilic / high-ammonia sets). Each scenario selects one via
`initial_states:`. The set as declared (with its `pH` when given) is written
in run.json -> calibration.state_variables.

### [`configs/influent.yaml`](../configs/influent.yaml) — what comes in

Defines the digester feed in one of two modes:

- **`dynamic`** — a CSV time series referenced via `file_path:`. Default:
  `data/influent/bsm2/daily_averages.csv` (BSM2 daily-averaged influent).
- **`constant`** — fixed values defined directly in the YAML.

Custom feedstocks (e.g. `pig_slurry`) are added as additional named
blocks in this file; scenarios then reference them by name via
`influent_mode:`. A dynamic CSV is copied into the run package
(`influent.csv`, sha256 in run.json); which row drives each interval is set
by `influent_hold` in `simulation.yaml`.

### [`configs/scenarios.yaml`](../configs/scenarios.yaml) — the user's entry point

The **main entry point**. A scenario is a self-contained recipe that
bundles:

- A description (`description:`, required: stored with the scenario in the
  STAMM registry).
- An initial-state set (`initial_states:`).
- An influent mode (`influent_mode:`).
- Parameter overrides (`parameter_overrides:`), the operating temperature
  `T_op` included.
- An optional horizon (`t_end:`).
- Optional hybrid hooks (`adm1_hybrid:`) — see [`hybrid.md`](hybrid.md).

A scenario is run with `python main.py --scenario <name>`;
`default_scenario:` is run without the option. The scenario name is its name
in the STAMM registry: other values need another name. Provided scenarios
include reference BSM2 (dynamic and constant), thermophilic conditions,
batch validation, hybrid demos (fictitious models, illustrative only), and a
pig-slurry feedstock; `python main.py --list-scenarios` lists them.

> This file defines the **simulation context**, not the full parameter set.

### [`configs/simulation.yaml`](../configs/simulation.yaml) — numerics, not biology

Numerical settings only — independent of the physical model:

- Solver selection (BDF / Radau / LSODA; explicit RK45 / DOP853 refused
  unless `allow_explicit: true`: the system is stiff)
- Tolerances (`rtol`, `atol`, `max_step`), restart at each influent row
  (`piecewise`)
- Which influent row drives each interval (`influent_hold`: start / end / legacy)
- Time horizon (`t_start`, `t_end`, fallback for constant-influent runs)
- Output time step and output options (results / runs folders, run package)
- The stamm-model-core settings (`mode: continuous`, `include_initial`)

This separation lets you tighten the solver without touching the
biophysical model and vice versa.

### [`configs/calibration.yaml`](../configs/calibration.yaml) — what is adjustable

> **Not used yet:** no script of this repository reads this file (neither `main.py` nor
> stamm-model-core). It documents the intended calibration set-up; the calibration itself is
> still to be written.

Defines the calibration framework:

- List of parameters to calibrate (with initial values and bounds)
- Observed variables (CSV with experimental data, target columns)
- Objective function (RMSE / MAE / NSE) with per-output weights
- Optimisation method (`differential_evolution`, `Nelder-Mead`, ...)
- Sequential calibration steps (e.g. hydrolysis first, then methanogenesis)

This file ensures a **clear distinction between fixed model parameters
and adjustable parameters**. The fixed values stay in
`parameters.yaml`; only the free parameters and their bounds live
here.

---

## How the files fit together at runtime

When `python main.py --scenario <name>` runs:

1. stamm-model-core reads and checks every file of `configs/` (one error
   message lists every problem).
2. **`scenarios.yaml`** gives the scenario: which `initial_states:` set,
   which `influent_mode:`, the `parameter_overrides:` and the
   `adm1_hybrid:` config.
3. **`parameters.yaml`** is the parameter base; the scenario's
   `parameter_overrides:` are layered on top.
4. **`initial_states.yaml`** is loaded for the named initial-state set;
   if a target `pH:` is specified, the strong-anion concentration is
   back-solved to make the charge balance close.
5. **`influent.yaml`** is loaded for the influent mode.
6. **`simulation.yaml`** gives the solver, horizon, output times and
   `influent_hold`.
7. `model.py` builds the reactor (constants from T_op) and wires the hybrid
   hooks if an `adm1_hybrid:` block is present (see [`hybrid.md`](hybrid.md)).
8. stamm-model-core integrates (one solver piece per influent interval),
   adds the acid-base / COD / biogas columns, writes
   `results/dynamic_out.csv` and the run package `runs/<run_id>.zip`
   (checked with the registry's rules); plots go to `results/figures/`.

`calibration.yaml` is not read by any script yet (see above): it is a
plan of the calibration, not an input of the simulation.

---

## References

1. Sadrmajd, P., Mannion, P., Howley, E., & Lens, P. N. L.
   *PyADM1: A Python Implementation of Anaerobic Digestion Model No. 1.*
   <https://doi.org/10.1101/2021.03.03.433746>
2. Rosen, C., & Jeppsson, U. (2006).
   *Aspects on ADM1 Implementation within the BSM2 Framework.*
   Lund University, Department of Industrial Electrical Engineering and Automation.
3. Batstone, D. J. *et al.* (2002).
   *Anaerobic Digestion Model No. 1 (ADM1).*
   IWA Scientific and Technical Report No. 13.
4. Yeghiazaryan, S., Capson-Tojo, G., Steyer, J.-P., et al. (2026).
   *Modeling Thermophilic Syntrophic VFA Oxidation Using Thermodynamic Principles.*
   <https://doi.org/10.1016/j.biortech.2026.134365>

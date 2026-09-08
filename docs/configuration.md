# Configuration files — design rationale and per-file reference

**Authors**
- Margaux Bonal — <margaux.bonal@inrae.fr>
- David Camilo Corrales — <David-Camilo.Corrales-Munoz@inrae.fr>

**Date:** April 2026

This document explains how the ADM1 configuration is split across the
six YAML files in [`../configs/`](../configs/), what each one is for,
what it should and shouldn't contain, and why it's organised this way.
Read this if you want to add a new scenario, calibrate parameters, or
extend the configuration.

---

## TL;DR — one file per question

The configuration follows a **separation of responsibilities**: each
file answers exactly one question about the simulation. Six files, six
questions, no overlap.

| Question | File | What lives here |
| --- | --- | --- |
| What are the reference ADM1 parameters? | [`configs/adm1_parameters.yaml`](../configs/adm1_parameters.yaml) | All intrinsic ADM1 constants — kinetic, stoichiometric, physico-chemical. **Independent of any specific simulation.** |
| What is the starting state of the digester? | [`configs/Initial_states.yaml`](../configs/Initial_states.yaml) | Named initial-state vectors (BSM2 reference, custom start-ups, ...). |
| What enters the digester? | [`configs/Influent.yaml`](../configs/Influent.yaml) | Influent definitions: dynamic CSV time series or constant values. |
| What do we want to test? | [`configs/Scenario.yaml`](../configs/Scenario.yaml) | Active-scenario selector + per-scenario overrides. **Main entry point for the user.** |
| How are the equations solved? | [`configs/Simulation.yaml`](../configs/Simulation.yaml) | ODE solver settings, time horizon, output options. |
| Which parameters are adjusted? | [`configs/Calibration.yaml`](../configs/Calibration.yaml) | Calibration framework: free parameters, bounds, objective function. |

This split is what lets you:

- **Compare simulation cases** without touching the model parameters
  (only swap scenarios).
- **Calibrate parameters** without entangling them with operating
  conditions (free parameters live in `Calibration.yaml`, not in
  `adm1_parameters.yaml`).
- **Switch operating regimes** (mesophilic ↔ thermophilic) without
  forking the parameter file (use `parameter_overrides:` in
  `Scenario.yaml`).
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

### [`configs/adm1_parameters.yaml`](../configs/adm1_parameters.yaml) — the model

The **central, stable parameter library** for ADM1. Values from
Rosen et al. (2006) BSM2 report and Batstone et al. (2002).

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
> `Scenario.yaml > parameter_overrides:`, not here.

### [`configs/Initial_states.yaml`](../configs/Initial_states.yaml) — where the digester starts

Named initial-state vectors. Each set defines the 38 ADM1 state
variables at `t = 0`:

- Dissolved states (`S_su`, `S_aa`, `S_fa`, `S_va`, `S_bu`, `S_pro`, `S_ac`, `S_h2`, `S_ch4`, `S_IC`, `S_IN`, `S_I`)
- Particulate states (`X_xc`, `X_ch`, `X_pr`, `X_li`, biomass `X_*`, `X_I`)
- Ionic states + pH (`S_cation`, `S_anion`, `S_H_ion`, `S_*_ion`, `S_co2`, `S_nh3`, `S_nh4_ion`)
- Gas-phase states (`S_gas_h2`, `S_gas_ch4`, `S_gas_co2`)

Multiple sets can coexist (e.g. `BSM2` reference, plus custom start-ups
or thermophilic / high-ammonia sets). Each scenario selects one via
`initial_states:`.

### [`configs/Influent.yaml`](../configs/Influent.yaml) — what comes in

Defines the digester feed in one of two modes:

- **`dynamic`** — a CSV time series referenced via `file_path:`. Default:
  `configs/daily_averages.csv` (BSM2 daily-averaged influent).
- **`constant`** — fixed values defined directly in the YAML.

Custom feedstocks (e.g. `pig_slurry`) are added as additional named
blocks in this file; scenarios then reference them by name via
`influent_mode:`.

### [`configs/Scenario.yaml`](../configs/Scenario.yaml) — the user's entry point

The **main entry point**. A scenario is a self-contained recipe that
bundles:

- An initial-state set (`initial_states:`).
- An influent mode (`influent_mode:`).
- Operating temperature (`T_op:`).
- Optional parameter overrides (`parameter_overrides:`).
- Optional hybrid hooks (`hybrid:`) — see [`hybrid.md`](hybrid.md).

The active scenario is chosen at the top of the file with
`active_scenario:`. Provided scenarios include reference BSM2 (dynamic
and constant), thermophilic conditions, batch validation, hybrid demos,
and a pig-slurry feedstock; see the README for the full list.

> This file defines the **simulation context**, not the full parameter set.

### [`configs/Simulation.yaml`](../configs/Simulation.yaml) — numerics, not biology

Numerical settings only — independent of the physical model:

- Solver selection (BDF / RK45 / Radau / LSODA / DOP853)
- Tolerances (`rtol`, `atol`, `max_step`)
- Time horizon (`t_start`, `t_end`, fallback for constant-influent runs)
- Output time step and output options
- Verbosity / progress display

This separation lets you tighten the solver without touching the
biophysical model and vice versa.

### [`configs/Calibration.yaml`](../configs/Calibration.yaml) — what is adjustable

Defines the calibration framework:

- List of parameters to calibrate (with initial values and bounds)
- Observed variables (CSV with experimental data, target columns)
- Objective function (RMSE / MAE / NSE) with per-output weights
- Optimisation method (`differential_evolution`, `Nelder-Mead`, ...)
- Sequential calibration steps (e.g. hydrolysis first, then methanogenesis)

This file ensures a **clear distinction between fixed model parameters
and adjustable parameters**. The fixed values stay in
`adm1_parameters.yaml`; only the free parameters and their bounds live
here.

---

## How the files fit together at runtime

When `python main.py` runs:

1. **`Scenario.yaml`** is read first → picks the active scenario.
2. The active scenario tells the loader which `initial_states:` set and
   which `influent_mode:` to use, and supplies any `parameter_overrides:`
   and `hybrid:` config.
3. **`adm1_parameters.yaml`** is loaded as the parameter base; the
   scenario's `parameter_overrides:` are layered on top.
4. **`Initial_states.yaml`** is loaded for the named initial-state set;
   if a target `pH:` is specified, the strong-anion concentration is
   back-solved to make the charge balance close.
5. **`Influent.yaml`** is loaded for the active influent mode.
6. **`Simulation.yaml`** is loaded for solver and output settings.
7. The reactor is built; if a `hybrid:` block is present, hooks are
   wired in (see [`hybrid.md`](hybrid.md)).
8. `solve_ivp` runs; results are written under
   `results/dynamic_out.csv` and (if enabled) plots under
   `results/figures/`.

`Calibration.yaml` is read only when calibration is run explicitly —
it's not consumed by the standard simulation entry point.

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

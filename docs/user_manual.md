# model-adm1 — short user manual

## 1. The mental model

```
configs/scenarios.yaml           ← THE SWITCHBOARD: `python main.py --scenario <name>`; each scenario
        │                          picks one entry from each catalogue below (+ optional overrides)
        ├── initial_states: ──►  configs/initial_states.yaml   (catalogue of t=0 digester states)
        ├── influent_mode:  ──►  configs/influent.yaml         (catalogue of feeds: CSV or constant)
        ├── parameter_overrides (T_op, q_ad, ...) ─► on top of configs/parameters.yaml (ADM1 constants)
        └── adm1_hybrid: use: [...] ─► models/*.yaml           (catalogue of ML plug-ins, optional)

configs/simulation.yaml          ← NUMERICS ONLY: solver, tolerances, horizon, output step, influent
                                   hold, files
configs/states.yaml, outputs.yaml, model_card.yaml
                                 ← the description stamm-model-core needs (states, CSV columns, identity)
```

You never edit Python to run a case. You (1) describe a feed / a start state once in the
catalogues, (2) combine them in a scenario, (3) `python main.py --scenario <name>`.
The run itself (solver loop, results, run package) is done by stamm-model-core (`model.py` tells
it how to run the reactor; see `architecture.md`).

## 2. Run

```bash
pip install -e ../stamm-model-core   # once (not on PyPI)
pip install -r requirements.txt
python main.py                       # the default scenario (BSM2_ringtest) → results/ + runs/<run_id>.zip
python main.py --scenario BSM2_steady_state
python main.py --list-scenarios      # every scenario with its description
python main.py --list-models         # hybrid plug-ins available in models/
python main.py --list-hooks          # what a plug-in may replace (Rho_1…Rho_19, I_5…I_12, I_nh3, dy/dt residual)
```

Other options: `--no-package`, `--runs-dir <folder>`, `--no-results`, `--no-plots`,
`--influent-hold start|end|legacy` (see §5), `--t-end <d>` (a shorter horizon for a quick check
only: do not register such a package under the scenario's name).

Console: one line with the scenario, influent, horizon and solver; a progress line every 30 s;
then the number of rows, the final pH and methane flow, the notes (warnings) of the run, the path of
the run package and its check with the registry's rules (`VALID run package`).

## 3. The switchboard — `configs/scenarios.yaml`

```yaml
default_scenario: BSM2_ringtest         # run when --scenario is not given

scenarios:
  BSM2_dynamic:                         # reference run (constant q_ad), validated vs MATLAB BSM2
    description: >                      # required: stored with the scenario in the STAMM registry
      Reference BSM2 simulation, daily dynamic influent.
    initial_states: BSM2                # key in initial_states.yaml
    influent_mode: dynamic              # key in influent.yaml
    parameter_overrides:                # any name from parameters.yaml
      T_op: {value: 308.15, units: "K"} # operating temperature (Henry & acid-base constants follow it)
      # q_ad: {value: 170, units: "m^3.d^-1"}
    # t_end: 200                        # optional: scenario-level horizon [d], overrides simulation.yaml
```

The scenario name is the name of the scenario in the STAMM registry: a scenario with other values
must get another name.

Provided scenarios:

| scenario | start state | feed | notes |
|---|---|---|---|
| `BSM2_dynamic` | BSM2 | daily BSM2 feed, constant q_ad | PyADM1-identical; ~1 min |
| `BSM2_steady_state` | BSM2_steady_init | constant BSM2 feed, q_ad 170 | benchmark A (steady state, 200 d); 2 s |
| `BSM2_ringtest` | BSM2 | daily BSM2 feed, **dynamic Q** | default; validation vs MATLAB; ~1 min |
| `BSM2_ringtest_15min` | BSM2 | 15-min BSM2 feed, dynamic Q | exact MATLAB reproduction; ~50 min |
| `BSM2_constant` | BSM2 | constant BSM2 feed | free steady-state runs |
| `thermophilic` | BSM2 | daily feed | T_op 328 K + K_I_nh3 override |
| `batch_validation` | BSM2 | constant, q_ad = 0 | batch kinetics |
| `pig_slurry_test` | BSM2 | constant pig slurry, q_ad = 35 | example of a custom feed |
| `hybrid_demo`, `hybrid_lr_demo`, `hybrid_lr_spec_demo` | BSM2 | daily feed | fictitious ML plug-ins, illustrative only — **not** pure ADM1; the two `lr` demos give results NOT valid (X_ch < 0) |
| `no_xc_demo` | BSM2 | constant feed without X_xc | `use_xc = 0` |
| `Tatiana_RM_lab_5L`, `Tatiana_RM_lab_5L_39C` | Tatiana_lab_5L | 5 L lab digester | her data are not in git (not published yet) |

To add a case: copy a block, rename it, change the keys (and the description), run it with
`--scenario`.

## 4. The catalogues

### 4.1 Feeds — `configs/influent.yaml`
Each top-level key is a feed you can reference with `influent_mode:`.

*Dynamic (time series):*
```yaml
my_plant:
  type: dynamic
  file_path: "data/influent/my_plant/influent.csv"   # columns: time [d] + the 26 ADM1 influent variables
  time_column: "time"                 # S_su … X_I in kgCOD/m3, S_IC/S_IN/S_cation/S_anion in kmol/m3
  flow_column: "Q"                    # OPTIONAL: feed flow [m3/d] per row; omit → constant q_ad
```
The simulation ends at the last row of the CSV (`t_end: null`), each row is held until the next
one (`influent_hold: start`, §5), and the integrator restarts at every row (`solver.piecewise: true`).
The CSV is copied into the run package (`influent.csv`) with its sha256 in run.json.

*Constant:*
```yaml
my_feed:
  type: constant
  pH:   {value: 7.2}                  # OPTIONAL: if given, S_anion is recomputed to hit this pH
  S_su: {value: 0.01, units: "kgCOD.m^-3"}
  ...                                 # all 26 variables must be present (0.0 is fine)
```
Duration for constant feeds: `time.t_end_default_constant` in `simulation.yaml` (default 30 d;
use ≥ 3–5 × HRT = V_liq/q_ad to reach steady state).

### 4.2 Start states — `configs/initial_states.yaml`
Each top-level key is a 38-variable digester state referenced with `initial_states:`.
`BSM2` is the Rosen & Jeppsson steady state (S_cation ≈ 0, pH 7.263) — the benchmark start.
Rules:
* All 38 variables must be present. The 9 algebraic ones (`S_H_ion`, VFA ions, `S_hco3_ion`,
  `S_co2`, `S_nh3`, `S_nh4_ion`) are **recomputed from the charge balance at start-up**, so their
  values only need to be plausible.
* Do **not** add a `pH:` key unless you want `S_anion` overwritten to force that pH
  (that is what broke the original BSM2 start state).
* For a start-up from an inoculum: copy `BSM2`, set the biomass `X_*` to your inoculum, keep
  the ions consistent (S_cation − S_anion sets the alkalinity, hence the pH).

### 4.3 Constants — `configs/parameters.yaml`
The ADM1/BSM2 parameter set at 35 °C, identical to PyADM1. Do not edit for a case study —
use `parameter_overrides:` in the scenario instead, so the reference stays intact.
`K_H_*`, `p_gas_h2o`, `K_w`, `K_a_co2`, `K_a_IN` are recomputed from `T_op` by the code; the
YAML values are the 35 °C reference only (run.json lists the recomputed ones with source "computed"). The reactor geometry (`V_liq`, `V_gas`) and default
`q_ad` live here too (override per scenario, e.g. `q_ad: {value: 35}`).

### 4.3b Disintegration switch — with or without the composite `X_xc`
`parameters.yaml → disintegration.use_xc` (override per scenario):

| `use_xc` | Feed characterised as | Biomass decay goes to | Reference |
|---|---|---|---|
| `1` (default) | `X_xc` (composite) + `k_dis` disintegration | `X_xc` → `k_dis` → `X_ch/X_pr/X_li/X_I/S_I` | ADM1 2002 / BSM2 — the validated set-up |
| `0` | `X_ch / X_pr / X_li / X_I` directly (`X_xc = 0`) | `f_*_xb` · (`S_I, X_ch, X_pr, X_li, X_I`) directly | Batstone et al. 2015 (no composite) |

Use `0` when the feed composition is measured (carbohydrates / proteins / lipids) and you calibrate
hydrolysis: with `X_xc` in the chain, `k_dis` and `k_hyd_*` are two first-order steps in series and are
not separately identifiable. With `0`, `X_xc` is still in the state vector but never reacts (it is
transported hydraulically only), so **keep `X_xc = 0` in the feed** or that COD is lost as inert.
The decay fractions `f_*_xb` default to `f_*_xc` (the convention of the RM_without_Xc code); they
must sum to 1 (checked at start-up). With `use_xc: 1` the equations reduce exactly to the validated
ones (`tests/test_xc_flag.py`). Scenarios: `no_xc_demo`, `Tatiana_RM_lab_5L`.

### 4.3c Gas-flow law — overpressure (BSM2) or head-space at P_atm
`parameters.yaml → gas_flow.gas_law_patm` (override per scenario):

| `gas_law_patm` | `q_gas` | Physical picture | Use for |
|---|---|---|---|
| `0` (default) | `k_p · (P_gas − P_atm)` | head-space slightly pressurised, gas leaves through an outlet with friction `k_p` | BSM2 / full-scale; the validated set-up |
| `1` | `R·T_op·V_liq·(ρT8/16 + ρT9/64 + ρT10) / (P_atm − p_H2O)` | head-space held at `P_atm`, all transferred gas leaves at once (`k_p` unused) | lab reactor vented to a gas counter; RM_without_Xc code (Batstone et al. 2002) |

`1` is the `k_p → ∞` limit of `0`. The two differ by the head-space overpressure: in BSM2
(`k_p = 5e4`, `P_gas ≈ 1.07 bar`) switching to `1` lowers the gas partial pressures and the dissolved
CH₄/CO₂ by ≈ 5 %, raises pH by 0.02 and free NH₃ by 5 % at steady state, but the biogas produced
(dry, normal conditions) changes by < 0.1 %. With `gas_law_patm: 0` the equations are exactly the
validated ones (`tests/test_gas_law.py`). Scenarios using `1`: `Tatiana_RM_lab_5L`, `Tatiana_RM_lab_5L_39C`.

### 4.4 ML plug-ins — `models/*.yaml` (optional)
```yaml
target: I_nh3                                   # or Rho_11, Rho_2, …, or residual
backend: callable                               # or npz artefact
artefact: examples/hybrid_inhibition_example.py:hill_nh3_inhibition
```
Activate in a scenario with `adm1_hybrid: {enabled: true, use: [nh3_hill, rho11_T_aware]}`; the
models used and the sha256 of their files go to run.json. The models in `models/` today are
fictitious examples (never tested on data). Details: `docs/hybrid.md`.

## 5. Numerics — `configs/simulation.yaml`
| key | default | meaning |
|---|---|---|
| `solver.method` | `BDF` | keep implicit (`BDF`, `Radau`, `LSODA`): S_h2 makes the system stiff |
| `solver.rtol / atol` | 1e-6 / 1e-8 | tighten for benchmarking, loosen for fast screening |
| `solver.max_step` | 1.0 d | upper bound; automatically capped at the influent step |
| `solver.piecewise` | true | restart at each influent row (recommended) |
| `influent_hold` | `start` | row held over each influent interval: `start` (a row holds until the next, best match of MATLAB), `end` (the next row, as the PyADM1 loop), `legacy` (before 2026-11-06); see validation.md §E |
| `mode`, `dt_out`, `t_start`, `include_initial` | continuous, 1.0, 0, true | stamm-model-core settings (the output times come from `time.dt_out`) |
| `time.t_end` | null | null → end of influent data (dynamic) / `t_end_default_constant` (constant) |
| `time.dt_out` | null | output resolution; null = same grid as the influent CSV (15-min feed → 26 881 rows, like PyADM1); a number forces that step |
| `output.*` | | results / runs folders, zip, code files hashed in run.json, `save_figures`, `show_figures`, `run_package.enabled` |

## 6. Outputs
`results/dynamic_out.csv`: `time` + the 38 states + `pH`, `charge_residual`, COD diagnostics and
biogas flows, in the order of `configs/outputs.yaml` (which documents every column).

`runs/<run_id>/` and `runs/<run_id>.zip`: the run package for the STAMM registry (Dynamic Models >
Register simulations), written by stamm-model-core: `dynamic_out.csv`, `run.json` (model, resolved
parameters, initial set, influent, hybrid, solver, warnings, columns) and `influent.csv` for a
dynamic influent; checked with the registry's rules after each run. See `run_package.md`.
Units: kgCOD/m³ for organics, kmol/m³ for S_IC, S_IN, ions. **VFA ions (`S_*_ion`) are in
kmol/m³** (PyADM1/MATLAB use kgCOD/m³: ×64, 112, 160, 208). Gas states `S_gas_*` are
concentrations; partial pressures p = S_gas·R·T (/16 for H₂, /64 for CH₄).

Biogas flows, all in m³/d (× 1e6 → mL/d), computed with the same law as the simulation:

| column | meaning |
|---|---|
| `q_gas` | wet gas at `T_op` and head-space pressure `P_gas` — the flow used in the ODEs |
| `q_gas_atm` | wet gas at `T_op` and `P_atm` = `q_gas · P_gas/P_atm` (the BSM2 report value: 2955.70 m³/d at steady state) |
| `q_gas_norm_dry` | dry gas at 0 °C, 1.01325 bar = `q_gas · (P_gas − p_H2O)/1.01325 · 273.15/T_op` — compare with a gas counter (NmL/d) |
| `q_ch4_norm_dry` | dry CH₄ at 0 °C, 1.01325 bar = `q_gas · p_CH4/1.01325 · 273.15/T_op` |
Figures: `results/figures/biogas.png`, `biomass.png`, `pH_alkalinity.png`.

## 7. Validate (do this after any change to `src/`)
```bash
# A. steady state (2 s)
python main.py --scenario BSM2_steady_state
python tests/benchmark_bsm2_steady.py --results results/dynamic_out.csv
# B. dynamic ring test (add --dynamic-q for BSM2_ringtest; BSM2_dynamic without it)
python main.py --scenario BSM2_ringtest
python tests/benchmark_bsm2.py --results results/dynamic_out.csv --dynamic-q \
       --reference tests/data/Matlabout_dyn.csv --outdir results/benchmark
```
`VERDICT: PASS` on both = still equivalent to the BSM2 references (test A also checks
`q_gas_atm` against the published 2955.70 m³/d). Switch tests (instantaneous):
`python tests/test_xc_flag.py` and `python tests/test_gas_law.py`; short runs on stamm-model-core:
`python tests/test_model.py`. See `docs/validation.md`.

### 7b. Cross-test of the no-composite variant (Tatiana's 5 L digester)
Her data are not in git (not published yet): this needs the local copy.
```bash
python main.py --scenario Tatiana_RM_lab_5L        # ~1 min
python tests/compare_tatiana_rm.py --results results/dynamic_out.csv   # → PASS (|avg| ≤ 1.1 %, nRMSE ≤ 6.1 %)
```
Compares 193 d of the 5 L lab digester (daily feed, dynamic flow, feed in X_ch/X_pr/X_li/X_I,
calibrated kinetics) with the trajectory of the independent RM_without_Xc code. See `docs/validation.md`.

## 8. Typical workflows
* **New feedstock, same digester:** add a block in `influent.yaml` → scenario with
  `initial_states: BSM2`, your `influent_mode`, `q_ad` override if the HRT changes → run.
* **Temperature study:** duplicate a scenario, change `T_op` (constants follow automatically).
* **Measured feed composition / solid substrate:** set `use_xc: 0` in `parameter_overrides`, put the
  feed in `X_ch/X_pr/X_li/X_I` with `X_xc = 0` (§4.3b), calibrate `k_hyd_*` — see `Tatiana_RM_lab_5L`.
* **Calibration:** keep `parameters.yaml` fixed; put the free parameters in
  `parameter_overrides`; `configs/calibration.yaml` documents bounds/objective (framework only,
  no optimiser is wired yet).
* **Hybrid ML:** write the callable in `examples/`, describe it in `models/x.yaml`, list it under
  `adm1_hybrid.use` — and check the identity version reproduces the benchmark.

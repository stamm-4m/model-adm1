# model-adm1 — short user manual

## 1. The mental model

```
configs/Scenario.yaml            ← THE SWITCHBOARD: one `active_scenario`, each scenario picks
        │                          one entry from each catalogue below (+ optional overrides)
        ├── initial_states: ──►  configs/Initial_states.yaml   (catalogue of t=0 digester states)
        ├── influent_mode:  ──►  configs/Influent.yaml         (catalogue of feeds: CSV or constant)
        ├── T_op / parameter_overrides ─► on top of configs/adm1_parameters.yaml (the ADM1 constants)
        └── hybrid: use: [...] ─► models/*.yaml                (catalogue of ML plug-ins, optional)

configs/Simulation.yaml          ← NUMERICS ONLY: solver, tolerances, horizon, output step, files
```

You never edit Python to run a case. You (1) describe a feed / a start state once in the
catalogues, (2) combine them in a scenario, (3) point `active_scenario` at it, (4) `python main.py`.

## 2. Run

```bash
pip install -r requirements.txt
python main.py                    # runs the active scenario → results/dynamic_out.csv + results/figures/
python main.py --list-models      # hybrid plug-ins available in models/
python main.py --list-hooks       # what a plug-in may replace (Rho_1…Rho_19, I_5…I_12, I_nh3, dy/dt residual)
```

Console: a banner with the scenario, reactor, feed and solver, then daily summaries (pH,
biogas, inhibitions, COD), then `Success: True`.

## 3. The switchboard — `configs/Scenario.yaml`

```yaml
active_scenario : BSM2_dynamic          # <── change only this line to switch cases

scenarios:
  BSM2_dynamic:                         # reference run (constant q_ad), validated vs MATLAB BSM2
    initial_states: BSM2                # key in Initial_states.yaml
    influent_mode: dynamic              # key in Influent.yaml
    T_op: {value: 308.15, units: "K"}   # operating temperature (Henry & acid-base constants follow it)
    parameter_overrides: {}             # any name from adm1_parameters.yaml, e.g. k_m_ac: {value: 10}
```

Provided scenarios:

| scenario | start state | feed | notes |
|---|---|---|---|
| `BSM2_dynamic` | BSM2 | daily BSM2 feed, constant q_ad | default; PyADM1-identical; ~2 min |
| `BSM2_ringtest` | BSM2 | daily BSM2 feed, **dynamic Q** | validation vs MATLAB; ~2 min |
| `BSM2_ringtest_15min` | BSM2 | 15-min BSM2 feed, dynamic Q | exact MATLAB reproduction; ~50 min |
| `BSM2_constant` | BSM2 | constant mean feed | steady-state runs |
| `thermophilic` | BSM2 | daily feed | T_op 328 K + K_I_nh3 override |
| `batch_validation` | BSM2 | constant, q_ad = 0 | batch kinetics |
| `pig_slurry_test` | BSM2 | constant pig slurry, q_ad = 35 | example of a custom feed |
| `hybrid_demo`, `hybrid_lr_demo`, `hybrid_lr_spec_demo` | BSM2 | daily feed | ML plug-ins active — **not** pure ADM1 |

To add a case: copy a block, rename it, change the keys, set `active_scenario` to it.

## 4. The catalogues

### 4.1 Feeds — `configs/Influent.yaml`
Each top-level key is a feed you can reference with `influent_mode:`.

*Dynamic (time series):*
```yaml
my_plant:
  type: dynamic
  file_path: "configs/my_plant.csv"   # columns: time [d] + the 26 ADM1 influent variables
  time_column: "time"                 # S_su … X_I in kgCOD/m3, S_IC/S_IN/S_cation/S_anion in kmol/m3
  flow_column: "Q"                    # OPTIONAL: feed flow [m3/d] per row; omit → constant q_ad
```
The simulation ends at the last row of the CSV (`t_end: null`), the feed is held constant
between rows, and the integrator restarts at every row (`solver.piecewise: true`).

*Constant:*
```yaml
my_feed:
  type: constant
  pH:   {value: 7.2}                  # OPTIONAL: if given, S_anion is recomputed to hit this pH
  S_su: {value: 0.01, units: "kgCOD.m^-3"}
  ...                                 # all 26 variables must be present (0.0 is fine)
```
Duration for constant feeds: `time.t_end_default_constant` in `Simulation.yaml` (default 30 d;
use ≥ 3–5 × HRT = V_liq/q_ad to reach steady state).

### 4.2 Start states — `configs/Initial_states.yaml`
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

### 4.3 Constants — `configs/adm1_parameters.yaml`
The ADM1/BSM2 parameter set at 35 °C, identical to PyADM1. Do not edit for a case study —
use `parameter_overrides:` in the scenario instead, so the reference stays intact.
`K_H_*`, `p_gas_h2o`, `K_w`, `K_a_co2`, `K_a_IN` are recomputed from `T_op` by the code; the
YAML values are the 35 °C reference only. The reactor geometry (`V_liq`, `V_gas`) and default
`q_ad` live here too (override per scenario, e.g. `q_ad: {value: 35}`).

### 4.4 ML plug-ins — `models/*.yaml` (optional)
```yaml
target: I_nh3                                   # or Rho_11, Rho_2, …, or residual
backend: callable                               # or npz artefact
artefact: examples/hybrid_inhibition_example.py:hill_nh3_inhibition
```
Activate in a scenario with `hybrid: {enabled: true, use: [nh3_hill, rho11_T_aware]}`.
Details: `docs/hybrid.md`.

## 5. Numerics — `configs/Simulation.yaml`
| key | default | meaning |
|---|---|---|
| `solver.method` | `BDF` | keep implicit (`BDF`, `Radau`, `LSODA`): S_h2 makes the system stiff |
| `solver.rtol / atol` | 1e-6 / 1e-8 | tighten for benchmarking, loosen for fast screening |
| `solver.max_step` | 1.0 d | upper bound; automatically capped at the influent step |
| `solver.piecewise` | true | restart at each influent row (recommended) |
| `time.t_end` | null | null → end of influent data (dynamic) / `t_end_default_constant` (constant) |
| `time.dt_out` | null | output resolution; null = same grid as the influent CSV (15-min feed → 26 881 rows, like PyADM1); a number forces that step |
| `output.*` | | output dir/file name, columns, `save_figures`, `show_figures` |

## 6. Outputs
`results/dynamic_out.csv`: `time` + the 38 states + `pH`, `charge_residual`, COD diagnostics.
Units: kgCOD/m³ for organics, kmol/m³ for S_IC, S_IN, ions. **VFA ions (`S_*_ion`) are in
kmol/m³** (PyADM1/MATLAB use kgCOD/m³: ×64, 112, 160, 208). Gas states `S_gas_*` are
concentrations; partial pressures p = S_gas·R·T (/16 for H₂, /64 for CH₄).
Figures: `results/figures/biogas.png`, `biomass.png`, `pH_alkalinity.png`.

## 7. Validate (do this after any change to `src/`)
```bash
python main.py                                    # active_scenario: BSM2_dynamic
python tests/benchmark_bsm2.py --results results/dynamic_out.csv \
       --reference tests/data/Matlabout_dyn.csv --outdir results/benchmark
```
`VERDICT: PASS` = still equivalent to the MATLAB BSM2 reference. Add `--dynamic-q` for the
`BSM2_ringtest*` scenarios. See `docs/validation.md`.

## 8. Typical workflows
* **New feedstock, same digester:** add a block in `Influent.yaml` → scenario with
  `initial_states: BSM2`, your `influent_mode`, `q_ad` override if the HRT changes → run.
* **Temperature study:** duplicate a scenario, change `T_op` (constants follow automatically).
* **Calibration:** keep `adm1_parameters.yaml` fixed; put the free parameters in
  `parameter_overrides`; `configs/Calibration.yaml` documents bounds/objective (framework only,
  no optimiser is wired yet).
* **Hybrid ML:** write the callable in `examples/`, describe it in `models/x.yaml`, list it under
  `hybrid.use` — and check the identity version reproduces the benchmark.

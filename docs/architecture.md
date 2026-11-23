# Architecture — how a run flows through the code

```mermaid
flowchart LR
    cfg[configs/*.yaml<br/>scenarios · parameters ·<br/>initial states · influent] --> sim
    inf[CSV influent<br/>time series] --> sim[ADM1 ODE<br/>BDF solver, restarted<br/>at each influent step]
    reg[models/*.yaml<br/>ML plug-ins, optional] -.-> sim
    sim --> csv[results/dynamic_out.csv<br/>full 38-state trajectory]
    sim --> pkg[runs/run_id.zip<br/>STAMM run package]
    sim --> plots[results/figures/<br/>biogas · biomass · pH/alkalinity]
    csv --> bench[tests/benchmark_bsm2.py<br/>vs MATLAB BSM2]

    classDef io fill:#eef6fb,stroke:#1a6e9e,color:#0b3a5b
    classDef core fill:#e6f7f2,stroke:#117a65,color:#0b5345
    class cfg,inf,csv,pkg,plots,reg,bench io
    class sim core
```

![architecture](images/architecture.png)

## Modules

| file | role |
|---|---|
| `main.py` | entry point: `--scenario <key>`; resolves the parameters, initial set, influent, output times and solver settings, then calls stamm-model-core `run_scenario` (results CSV, run package checked with the registry's rules) and draws the figures |
| `model.py` | `ADM1(OdeModel)`: how stamm-model-core runs the reactor — initial state (hybrid hooks wired, acid-base re-projection), K_w / K_a_co2 / K_a_IN recomputed from T_op, solver restarts at the influent rows, the influent row held over each piece (`influent_hold`), acid-base / pH / COD / biogas columns after the run, warning on negative concentrations |
| `initial_states.py` | `InitialState`: 38-variable y0 from `initial_states.yaml` (canonical `STATE_VARIABLES` order) |
| `src/parameters.py` | `ADM1Parameters`: flattens `parameters.yaml`, applies the scenario's `parameter_overrides` (T_op included); `from_values()` takes parameters already resolved by stamm-model-core |
| `src/influent.py` | `Influent`: dynamic (CSV, optional `flow_column`) or constant feed; `get(step)` → `{..._in}` dict |
| `src/reactor.py` | `ADM1Reactor`: inhibitions, 19 process rates, gas transfer, 38 mass balances; T-dependent constants; hybrid hook points |
| `src/acid_base.py` | charge-balance solver (pH, HCO₃⁻, CO₂, NH₃, NH₄⁺, VFA ions) called inside every RHS evaluation; COD diagnostics |
| `src/registry.py`, `src/hybrid.py` | load `models/*.yaml`, resolve backends, attach overrides to the reactor |
| `plots/` | the three diagnostic figures |
| `tests/benchmark_bsm2.py` | BSM2 ring test against `tests/data/Matlabout_dyn.csv` |

## Numerical formulation
* 29 dynamic states integrated (26 liquid + 3 gas); the 9 acid-base states are algebraic and
  recomputed at every RHS call (DAE formulation of Rosen & Jeppsson 2006).
* S_h2 is integrated (stiff) → implicit solver required (`BDF` default).
* The feed is a zero-order hold; the integrator restarts at every influent row and the row of each
  interval is fixed for the whole interval (`influent_hold: start`, docs/validation.md section E).
* Constants that depend on temperature (Henry, p_H2O, K_w, K_a_co2, K_a_IN) are recomputed
  from `T_op` at reactor construction.

The loop itself (pieces, output rows, run package) is stamm-model-core's continuous mode.
Images in `images/` are rendered by `scripts/render_diagrams.py` (drawn before the stamm-model-core
port: they do not show `model.py` and the run package).

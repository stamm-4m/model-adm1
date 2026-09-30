# Class diagram — model-adm1 from a software point of view

Five small classes, one orchestrating script, and a bag of pure functions. No inheritance anywhere;
everything is composed in `main.py`. Configuration (YAML) is the source of truth; the classes are thin
readers around it, and `ADM1Reactor` is the only object with real logic (the 38 ODEs).

```mermaid
classDiagram
    direction LR

    class main {
        <<script>>
        +main()
        -ADM1_wrapper(t, y) dydt
        -get_influent_for_time(t) dict
        -build_time_vector()
        -print_startup_banner()
    }

    class ADM1Parameters {
        +params : dict
        +params_file : str
        +scenarios_file : str
        +__getattr__(name) float
        +get(name, default) float
        -_load()
        -_load_scenario_overrides() dict
        -_flatten(yaml) dict
    }

    class InitialState {
        +state_dict : dict
        +get_vector() ndarray
        +get_dict() dict
        -_load()
    }

    class Influent {
        +mode : str
        +get(step) dict
        +get_time() ndarray
        -_data : DataFrame
        -_constant_values : dict
        -_flow_col : str
    }

    class ADM1Reactor {
        +param : ADM1Parameters
        +influent_state : dict
        +use_xc : bool
        +gas_law_patm : bool
        +f_sI_xb f_ch_xb f_pr_xb f_li_xb f_xI_xb
        +K_H_co2 K_H_ch4 K_H_h2 p_gas_h2o
        +K_pH_aa nn_aa K_pH_ac n_ac K_pH_h2 n_h2
        +rate_overrides : dict
        +inhibition_overrides : dict
        +residual_correction : callable
        +ADM1_ODE(t, y) dydt
        +compute_inhibitions(state) dict
        +compute_biochemical_rates(state, inhib) dict
        +compute_gas_transfer(state) dict
        +mass_balances(state, rho, gas) list
        +get_process_summary() dict
        +unpack_state(y) dict
        +reduce_to_dynamic_state(y38) y29
        +expand_dynamic_state(y29) y38
    }

    class acid_base {
        <<module>>
        +compute_acid_base_equilibrium(state, param) dict
        +compute_required_strong_ion_for_pH(state, param, pH) float
        +compute_total_cod(state, param) dict
        -_charge_balance(H, state, param)
    }

    class ModelEntry {
        <<dataclass>>
        +name : str
        +target : str
        +backend : str
        +artefact : str
        +inputs : list
    }

    class registry {
        <<module>>
        +load_registry(models_dir) dict
        +format_registry(registry) str
        +format_hooks() str
    }

    class hybrid {
        <<module>>
        +apply_hybrid_config(reactor, cfg, registry) dict
        +load_callable(spec) callable
        -_compile_entry(entry) callable
        -_load_artefact(backend, path)
    }

    class plots {
        <<module>>
        +plot_biogas(df, param)
        +plot_biomass(df)
        +plot_pH_alkalinity(df, param)
    }

    class solve_ivp {
        <<scipy>>
    }

    class configs {
        <<YAML>>
        Scenario.yaml
        adm1_parameters.yaml
        Initial_states.yaml
        Influent.yaml
        Simulation.yaml
    }

    main --> ADM1Parameters : builds
    main --> InitialState : builds
    main --> Influent : builds
    main --> ADM1Reactor : builds
    main --> registry : load_registry()
    main --> hybrid : apply_hybrid_config()
    main --> solve_ivp : per influent step
    main --> plots : after the run
    solve_ivp ..> main : calls ADM1_wrapper(t, y)
    ADM1Reactor o-- ADM1Parameters : param
    ADM1Reactor ..> acid_base : every RHS call
    ADM1Reactor ..> Influent : influent_state (dict injected by main)
    registry ..> ModelEntry : creates
    hybrid ..> ModelEntry : compiles
    hybrid ..> ADM1Reactor : sets overrides and residual_correction
    InitialState ..> acid_base : (via main) ions at t=0
    Influent ..> acid_base : constant feed with pH key
    ADM1Parameters ..> configs : adm1_parameters.yaml + Scenario.yaml
    InitialState ..> configs : Initial_states.yaml + Scenario.yaml
    Influent ..> configs : Influent.yaml + Scenario.yaml + CSV
    registry ..> configs : models yaml files

```

## Reading guide

* **`Scenario.yaml` is the switchboard.** `ADM1Parameters`, `InitialState` and `Influent` each open it
  only to read `active_scenario` and pick their own block (`parameter_overrides` / `initial_states` /
  `influent_mode`). They never talk to each other.
* **`ADM1Parameters` is a flat `dict` with attribute access** (`param.k_dis`, `param.use_xc`). Priority:
  scenario `parameter_overrides` > scenario `T_op` > `adm1_parameters.yaml`. The reactor also *writes*
  into it (`K_w`, `K_a_co2`, `K_a_IN` recomputed from `T_op`; `q_ad` updated per feed row when the
  influent has a flow column).
* **`ADM1Reactor` is stateless between calls** except for the `influent_state` dict that `main` injects
  before every RHS evaluation and the `_last_*` caches used for logging. `ADM1_ODE` is the pipeline
  `acid_base → inhibitions → rates → gas → mass_balances`, with the three hybrid hook points in between.
* **Two state vectors.** The solver integrates 29 dynamic states (`y29`); the reactor works on the full
  38 (`y38`), the 9 acid–base states being recomputed algebraically at each call
  (`reduce_to_dynamic_state` / `expand_dynamic_state` convert).
* **`use_xc`** lives in `ADM1Parameters` (YAML) and is read once at reactor construction into
  `ADM1Reactor.use_xc`; it changes `Rho_1` and the decay routing in `mass_balances`.
* **`gas_law_patm`** is read the same way into `ADM1Reactor.gas_law_patm`; it only changes `q_gas` in
  `compute_gas_transfer` (and the same law in `compute_gas_flow_rate` / `compute_gas_outputs`, which
  produce the `q_gas*` CSV columns and the biogas plot).
* **Hybrid layer** = `registry` (reads `models/*.yaml` → `ModelEntry`) + `hybrid` (turns an entry into a
  Python callable and plugs it into `rate_overrides`, `inhibition_overrides` or `residual_correction`).
  With no `hybrid:` block in the scenario the reactor is pure ADM1.

## One simulation, as a sequence

```mermaid
sequenceDiagram
    autonumber
    participant M as main.py
    participant P as ADM1Parameters
    participant S as InitialState
    participant I as Influent
    participant R as ADM1Reactor
    participant A as acid_base
    participant H as registry / hybrid
    participant O as scipy.solve_ivp

    M->>P: ADM1Parameters()  (YAML + scenario overrides)
    M->>S: InitialState()    (y0, 38 values)
    M->>I: Influent()        (CSV or constant block)
    M->>R: ADM1Reactor(param) - T constants, use_xc, f_xb
    M->>H: load_registry() + apply_hybrid_config(reactor, hybrid cfg)
    H-->>R: rate_overrides / inhibition_overrides / residual_correction
    M->>A: compute_acid_base_equilibrium(y0) - consistent ions at t=0
    M->>R: reduce_to_dynamic_state(y0_38) to y0_29
    loop each influent row (solver.piecewise = true)
        M->>O: solve_ivp(ADM1_wrapper, (t_i, t_i+1), y, BDF)
        loop each RHS evaluation
            O->>M: ADM1_wrapper(t, y29)
            M->>R: influent_state = influent.get(row), param.q_ad = Q(t)
            M->>R: ADM1_ODE(t, y29)
            R->>R: expand_dynamic_state to y38
            R->>A: compute_acid_base_equilibrium(state) - pH, ions
            R->>R: compute_inhibitions, compute_biochemical_rates, compute_gas_transfer
            R->>R: mass_balances (38 eq.) + residual_correction
            R-->>O: dydt (29)
        end
    end
    M->>R: expand + acid_base on every output row (38 states + pH)
    M->>M: write results/dynamic_out.csv and figures
```

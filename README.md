# model-adm1 — Anaerobic Digestion Model No. 1 in Python

A modular Python implementation of **ADM1** (Batstone et al., 2002) in its BSM2 form
(Rosen & Jeppsson, 2006), refactored from [PyADM1](https://github.com/CaptainFerMag/PyADM1),
configured entirely through YAML, and with optional plug points for ML models (hybrid mode).

**Validated:** reproduces the published BSM2 steady state to 8·10⁻⁷ (PyADM1: 2·10⁻⁵) and the
MATLAB/Simulink BSM2 dynamic trajectory to ≤ 0.04 % on all 38 states (see [docs/validation.md](docs/validation.md)).

**Authors:** Margaux Bonal — <margaux.bonal@inrae.fr> · David Camilo Corrales — <David-Camilo.Corrales-Munoz@inrae.fr> (INRAE / TBI)

---

## 1. Install

Python ≥ 3.10.

```bash
git clone https://github.com/stamm-4m/model-adm1.git
cd model-adm1
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy scipy pandas matplotlib pyyaml
```

## 2. Run

```bash
python main.py
```

Runs the scenario named in `configs/Scenario.yaml` (`active_scenario`, default `BSM2_dynamic`:
the BSM2 reference digester, 280 days of daily-averaged influent, ~2 min) and writes

```
results/dynamic_out.csv          time + 38 states + pH, charge residual, COD diagnostics
results/figures/*.png            biogas · biomass · pH/alkalinity
```

Other commands:

```bash
python main.py --list-models       # ML plug-ins registered in models/
python main.py --list-hooks        # what a plug-in may replace (Rho_1…Rho_19, I_5…I_12, I_nh3, dy/dt residual)
```

## 3. Configure — one switchboard, four catalogues, one numerics file

```
configs/Scenario.yaml            ← SWITCHBOARD: `active_scenario: <name>`; each scenario picks
        │                          one entry from each catalogue (+ optional overrides)
        ├── initial_states: ──►  configs/Initial_states.yaml   digester state at t = 0 (38 variables)
        ├── influent_mode:  ──►  configs/Influent.yaml         feeds: CSV time series or constant values
        ├── T_op, parameter_overrides ─► configs/adm1_parameters.yaml   ADM1/BSM2 constants (do not edit; override)
        └── hybrid.use: [...] ──►  models/*.yaml                ML plug-ins (optional)

configs/Simulation.yaml          ← NUMERICS: solver, tolerances, horizon, output step, file names
```

To run another case, change **one line**:

```yaml
# configs/Scenario.yaml
active_scenario : BSM2_ringtest        # e.g. the run with dynamic feed flow Q(t)
```

To create a case: add a feed block to `Influent.yaml` and/or a start state to `Initial_states.yaml`,
then combine them in a new scenario block:

```yaml
  my_case:
    initial_states: BSM2                # key in Initial_states.yaml
    influent_mode: my_plant             # key in Influent.yaml
    T_op: {value: 308.15, units: "K"}
    parameter_overrides: {q_ad: {value: 120.0}}
```

Provided scenarios: `BSM2_dynamic` (default), `BSM2_steady_state`, `BSM2_ringtest`, `BSM2_ringtest_15min`,
`BSM2_constant`, `thermophilic`, `batch_validation`, `pig_slurry_test`, and three `hybrid_*` demos
(not pure ADM1). Everything above in detail — templates, pitfalls, units, workflows — is in
**[docs/user_manual.md](docs/user_manual.md)**.

## 4. Validate

Two BSM2 benchmarks (Rosen & Jeppsson 2006), the same ones PyADM1 was validated with:

```bash
# A. steady-state test (2 s): final state vs the published BSM2 steady state, 38 states, 14 digits
#    configs/Scenario.yaml → active_scenario : BSM2_steady_state
python main.py
python tests/benchmark_bsm2_steady.py --results results/dynamic_out.csv        # → max |err| 8e-7, PASS

# B. dynamic ring test (2 min): 280 d of BSM2 influent vs the MATLAB/Simulink trajectory
#    configs/Scenario.yaml → active_scenario : BSM2_ringtest   (or BSM2_ringtest_15min, 50 min, exact)
python main.py
python tests/benchmark_bsm2.py --dynamic-q --results results/dynamic_out.csv \
       --reference tests/data/Matlabout_dyn.csv --outdir results/benchmark      # → PASS
```

Run both after any change to `src/`; a non-zero exit code means the equations changed. Plus
`python tests/test_gas_law.py` for the gas-flow law switch (`gas_law_patm`), `python tests/test_xc_flag.py` for the disintegration switch (`use_xc`, with/without the composite
`X_xc` — Batstone et al. 2015) and `tests/compare_tatiana_rm.py` for its cross-test on a 5 L lab
digester (`Tatiana_RM_lab_5L`); see [docs/validation.md](docs/validation.md) §C.
Protocol, metrics and expected numbers: [docs/validation.md](docs/validation.md).

## 5. Hybrid mode (optional)

Replace any process rate `Rho_1…Rho_19`, any inhibition factor `I_5…I_12, I_nh3`, or add a
residual correction on `dy/dt` with your own model, without touching `src/`: describe the model
in a `models/<name>.yaml` (target, backend `callable` / `linear_lstsq` / `sklearn`, artefact) and
list it in a scenario under `hybrid: {enabled: true, use: [<name>]}`.
Guide: [docs/hybrid.md](docs/hybrid.md) · recipes: [models/README.md](models/README.md) ·
worked examples: [examples/README.md](examples/README.md).

## 6. Project structure

```
model-adm1/
├── main.py                  entry point (load configs → integrate → CSV + figures)
├── initial_states.py        38-state initial vector loader
├── configs/                 Scenario.yaml (switchboard) · Influent.yaml · Initial_states.yaml ·
│                            adm1_parameters.yaml · Simulation.yaml · Calibration.yaml · influent CSVs
├── src/                     reactor.py (ODEs) · acid_base.py (pH/DAE) · parameters.py · influent.py ·
│                            hybrid.py · registry.py
├── models/                  hybrid-model registry (one YAML per model)
├── examples/                hybrid plug-in examples
├── plots/                   diagnostic figures
├── tests/                   benchmark_bsm2.py · benchmark_bsm2_steady.py · data/ (BSM2 references)
├── results/                 outputs
└── docs/                    documentation — start at docs/README.md
```

## 7. Documentation

| I want to … | read |
|---|---|
| run and configure the model | [docs/user_manual.md](docs/user_manual.md) |
| know how (and how well) it is validated | [docs/validation.md](docs/validation.md) · [full report](docs/validation_report_2026-09-08.md) |
| understand the biology / the 38-state ODE (CS framing) | [docs/adm1_biology.md](docs/adm1_biology.md) |
| understand the configuration design | [docs/configuration.md](docs/configuration.md) |
| see how a run flows through the code | [docs/architecture.md](docs/architecture.md) |
| plug in an ML model | [docs/hybrid.md](docs/hybrid.md) |

## 8. References and licence

* Batstone D.J. et al. (2002). *Anaerobic Digestion Model No. 1 (ADM1)*. IWA STR No. 13.
* Rosen C., Jeppsson U. (2006). *Aspects on ADM1 implementation within the BSM2 framework*. Lund University.
* Sadrimajd P. et al. (2021). *PyADM1: a Python implementation of ADM1*. bioRxiv 10.1101/2021.03.03.433746.

Licence: [Apache 2.0](LICENSE). Reference data in `tests/data/` from PyADM1 (MIT).

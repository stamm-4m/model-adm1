# model-adm1 — Anaerobic Digestion Model No. 1 in Python

A modular Python implementation of **ADM1** (Batstone et al., 2002) in its BSM2 form
(Rosen & Jeppsson, 2006), refactored from [PyADM1](https://github.com/CaptainFerMag/PyADM1),
configured entirely through YAML, and with optional plug points for ML models (hybrid mode).

**Validated:** reproduces the published BSM2 steady state to 8·10⁻⁷ (PyADM1: 2·10⁻⁵) and the
MATLAB/Simulink BSM2 dynamic trajectory to ≤ 0.04 % on all 38 states (see [docs/validation.md](docs/validation.md)).

Since 2026-11-06 it runs on [stamm-model-core](https://forge.inrae.fr/stamm/models/stamm-model-core),
the STAMM modelling package: `model.py` (`ADM1(OdeModel)`) tells the core how the previous
`main.py` ran the reactor; the core integrates, writes the results and the run package and checks
it with the registry's rules. The equations (`src/`) are unchanged and the trajectories are
identical to those of the previous code (`tools/compare_port.py`).

**Authors:** Margaux Bonal — <margaux.bonal@inrae.fr> · David Camilo Corrales — <David-Camilo.Corrales-Munoz@inrae.fr> (INRAE / TBI)

---

## 1. Install

Python ≥ 3.10.

```bash
git clone https://github.com/stamm-4m/model-adm1.git
cd model-adm1
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ../stamm-model-core  # or: pip install git+https://forge.inrae.fr/stamm/models/stamm-model-core.git
pip install -r requirements.txt    # numpy scipy pandas matplotlib pyyaml stamm-model-core
# stamm-model-core >= 0.2.2 is required: with 0.2.1 influent_hold is silently ignored
# (the run behaves as `legacy` while run.json records `start`).
```

## 2. Run

```bash
python main.py                              # default scenario: BSM2_ringtest
python main.py --scenario BSM2_dynamic      # any key of configs/scenarios.yaml
python main.py --list-scenarios
```

Runs one scenario of `configs/scenarios.yaml` (`default_scenario`, `BSM2_ringtest`: the BSM2
reference digester, daily-averaged influent with the dynamic flow Q(t), ~2 min) and writes

```
results/dynamic_out.csv          time + 38 states + pH, charge residual, COD diagnostics, biogas flows
results/figures/*.png            biogas · biomass · pH/alkalinity
runs/<run_id>/                   run package for the STAMM registry (never overwritten):
    dynamic_out.csv              same CSV as above
    run.json                     model, resolved parameters, initial states, influent, hybrid, solver, columns
    influent.csv                 copy of the influent (dynamic influent only)
runs/<run_id>.zip                the same files, ready to upload
```

The package is written by stamm-model-core (schema 1.0 of the STAMM registry);
`output.run_package.enabled` in `configs/simulation.yaml` (or `--no-package`) turns it off.
[docs/run_package.md](docs/run_package.md) describes `run.json` field by field.

Other commands:

```bash
python main.py --list-models       # ML plug-ins registered in models/
python main.py --list-hooks        # what a plug-in may replace (Rho_1…Rho_19, I_5…I_12, I_nh3, dy/dt residual)
```

## 3. Configure — one switchboard, four catalogues, one numerics file

```
configs/scenarios.yaml           ← SWITCHBOARD: `--scenario <name>`; each scenario picks
        │                          one entry from each catalogue (+ optional overrides)
        ├── initial_states: ──►  configs/initial_states.yaml   digester state at t = 0 (38 variables)
        ├── influent_mode:  ──►  configs/influent.yaml         feeds: CSV time series or constant values
        ├── parameter_overrides (T_op, ...) ─► configs/parameters.yaml   ADM1/BSM2 constants (do not edit; override)
        └── adm1_hybrid.use: [...] ─► models/*.yaml             ML plug-ins (optional)

configs/simulation.yaml          ← NUMERICS: solver, tolerances, horizon, output step, file names
configs/states.yaml, outputs.yaml, model_card.yaml   ← the description stamm-model-core needs
```

To run another case: `python main.py --scenario <name>`.

To create a case: add a feed block to `influent.yaml` and/or a start state to `initial_states.yaml`,
then combine them in a new scenario block:

```yaml
  my_case:
    initial_states: BSM2                # key in initial_states.yaml
    influent_mode: my_plant             # key in influent.yaml
    description: My plant, 35 C        # required: stored with the scenario in the registry
    parameter_overrides:
      T_op: {value: 308.15, units: "K"}
      q_ad: {value: 120.0, units: "m^3.d^-1"}
```

Provided scenarios: `BSM2_ringtest` (default), `BSM2_dynamic`, `BSM2_steady_state`, `BSM2_ringtest_15min`,
`BSM2_constant`, `thermophilic`, `batch_validation`, `pig_slurry_test`, `no_xc_demo`, and three `hybrid_*` demos
(illustrative, not pure ADM1). `Tatiana_RM_lab_5L` and `Tatiana_RM_lab_5L_39C` are also defined, but their
lab data are not published yet (see `.gitignore`); `python main.py --list-scenarios` lists them all. Everything above in detail — templates, pitfalls, units, workflows — is in
**[docs/user_manual.md](docs/user_manual.md)**.

## 4. Validate

Two BSM2 benchmarks (Rosen & Jeppsson 2006), the same ones PyADM1 was validated with:

```bash
# A. steady-state test (2 s): final state vs the published BSM2 steady state, 38 states, 14 digits
python main.py --scenario BSM2_steady_state
python tests/benchmark_bsm2_steady.py --results results/dynamic_out.csv        # → max |err| 8e-7, PASS

# B. dynamic ring test (2 min): 280 d of BSM2 influent vs the MATLAB/Simulink trajectory
#    (or --scenario BSM2_ringtest_15min, 50 min, exact)
python main.py --scenario BSM2_ringtest
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
list it in a scenario under `adm1_hybrid: {enabled: true, use: [<name>]}` (the files and their sha256
are written in run.json -> calibration.hybrid).
Guide: [docs/hybrid.md](docs/hybrid.md) · recipes: [models/README.md](models/README.md) ·
worked examples: [examples/README.md](examples/README.md).

## 6. Project structure

```
model-adm1/
├── main.py                  entry point (scenario → stamm-model-core run → CSV, run package, figures)
├── model.py                 ADM1(OdeModel): how stamm-model-core runs the reactor (hooks)
├── initial_states.py        38-state initial vector loader
├── configs/                 scenarios.yaml (switchboard) · influent.yaml · initial_states.yaml ·
│                            parameters.yaml · simulation.yaml · calibration.yaml · model_card.yaml ·
│                            states.yaml · outputs.yaml   (configuration only, YAML)
├── data/influent/<source>/  influent time series (CSV), one folder per data source — see data/README.md
├── src/                     reactor.py (ODEs) · acid_base.py (pH/DAE) · parameters.py · influent.py ·
│                            hybrid.py · registry.py
├── models/                  hybrid-model registry (one YAML per model)
├── examples/                hybrid plug-in examples
├── plots/                   diagnostic figures
├── tests/                   test_model.py · benchmark_bsm2.py · benchmark_bsm2_steady.py · data/ (BSM2 references)
├── tools/                   compare_port.py (stamm-model-core port vs the previous code)
├── results/                 outputs (overwritten at each run)
├── runs/                    one run package per run for STAMM (never overwritten) — docs/run_package.md
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

Licence: [Apache 2.0](LICENSE): the code of this repository. It runs on stamm-model-core, which is
AGPL-3.0-only: a distribution of model-adm1 together with stamm-model-core, or a service built on
both, is subject to the AGPL-3.0 as a whole. Reference data in `tests/data/` from PyADM1 (MIT).

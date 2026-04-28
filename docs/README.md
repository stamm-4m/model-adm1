# Documentation

Extended documentation for the ADM1 reactor simulation.

| File | What it covers |
| ---- | -------------- |
| [adm1_biology.md](adm1_biology.md) | The biology of ADM1 (anaerobic digestion) explained for computer scientists. Read this first if you have a CS background and want to understand *what the model is actually simulating*. |
| [configuration.md](configuration.md) | The six YAML configuration files: design rationale (why parameters are split this way) and a per-file reference (what each file should and shouldn't contain). |
| [hybrid.md](hybrid.md) | How to plug your own ML / regression / lookup callables into ADM1 (Tier 1: rate or inhibition replacement; Tier 2: UDE-style residual correction). YAML-driven, no code changes required. |

More documents will be added here over time (e.g. configuration reference,
calibration workflow, numerical-solver notes). If you write a new doc, list
it in the table above and link to it from the top-level [../README.md](../README.md).

## Authors

- Margaux Bonal — <margaux.bonal@inrae.fr>
- David Camilo Corrales — <David-Camilo.Corrales-Munoz@inrae.fr>

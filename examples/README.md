# Hybrid-mode examples

One worked example per plug-point type. Each is registered as a model
under [`../models/`](../models/) and wired into a demo scenario.

> **Illustrative only.** The models of these examples are fictitious: none was trained or tested on
> real data. They show how to plug a model into ADM1, not a validated hybrid ADM1. Two of them
> (`hybrid_lr_demo`, `hybrid_lr_spec_demo`) even drive X_ch negative: their runs are flagged
> "results NOT valid" in run.json -> run.status.warnings.

```mermaid
flowchart LR
    py["examples/<br/>foo_example.py"] --> yaml["models/<br/>foo.yaml"] --> sc["scenarios.yaml<br/>adm1_hybrid.use: [foo]"]
    classDef code fill:#e6f7f2,stroke:#117a65,color:#0b5345
    classDef yaml fill:#fff4e6,stroke:#cc6600,color:#663300
    class py code
    class yaml,sc yaml
```

| Example                                                                                       | Tier | Plug point          | Registered as                                            |
| --------------------------------------------------------------------------------------------- | ---- | ------------------- | -------------------------------------------------------- |
| [hybrid_rate_example.py](hybrid_rate_example.py)                                              | 1    | `Rho_11`            | [`rho11_T_aware`](../models/rho11_T_aware.yaml)          |
| [hybrid_inhibition_example.py](hybrid_inhibition_example.py)                                  | 1    | `I_nh3`             | [`nh3_hill`](../models/nh3_hill.yaml)                    |
| [hybrid_residual_example.py](hybrid_residual_example.py)                                      | 2    | `residual:S_ch4`    | [`methane_bias`](../models/methane_bias.yaml)            |
| [hybrid_linear_regression_example.py](hybrid_linear_regression_example.py)                    | 1    | `Rho_2` (ML)        | [`rho2_lr_module`](../models/rho2_lr_module.yaml)        |

The committed [`rho2_lr_synthetic`](../models/rho2_lr_synthetic.yaml) uses
the same predictions but loaded as a saved `.npz` (`backend: linear_lstsq`)
— useful for A/B comparison between the two load paths.

## Demo scenarios

| Scenario              | Activates                                                |
| --------------------- | -------------------------------------------------------- |
| `hybrid_demo`         | `rho11_T_aware` + `nh3_hill` + `methane_bias`            |
| `hybrid_lr_demo`      | `rho2_lr_module` (Python callable LR)                    |
| `hybrid_lr_spec_demo` | `rho2_lr_synthetic` (saved `.npz` LR)                    |

```bash
python main.py --scenario hybrid_demo
```

## Writing your own

Recipe + signatures: [`../models/README.md`](../models/README.md).
Validating wiring: register the `zero_residual` no-op from
[`hybrid_residual_example.py`](hybrid_residual_example.py) and confirm
the output is bit-for-bit identical to pure ADM1.

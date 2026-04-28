# Hybrid-mode examples

Each file here is a **drop-in replacement** for one part of the ADM1 model.
You enable any combination of them in [`configs/Scenario.yaml`](../configs/Scenario.yaml)
under your scenario's `hybrid:` block — no Python changes required.

See the full guide in [`docs/hybrid.md`](../docs/hybrid.md).

## What each example shows

| File | Tier | What it overrides | Pattern |
| ---- | ---- | ---- | ---- |
| [hybrid_rate_example.py](hybrid_rate_example.py) | 1 | `Rho_11` (acetoclastic methanogenesis) | classical Monod × Gaussian temperature factor — illustrates "physics × learned correction" |
| [hybrid_inhibition_example.py](hybrid_inhibition_example.py) | 1 | `I_nh3` (free-ammonia inhibition) | Hill sigmoid with tunable exponent — illustrates a parametric replacement |
| [hybrid_residual_example.py](hybrid_residual_example.py) | 2 | residual on `dy/dt` | constant CH₄ bias — illustrates UDE-style additive correction |
| [hybrid_linear_regression_example.py](hybrid_linear_regression_example.py) | 1 | `Rho_2` (carbohydrate hydrolysis) | **real ML model**: linear regression fitted on synthetic data via `np.linalg.lstsq` — full *train offline → predict online* workflow in one file |

For loading a saved model artefact through the simulator's plug-and-play
hooks, see the **save recipes** in [`../models/README.md`](../models/README.md).
The simulator is intentionally a loader-only target — train with whatever
ML stack you already use, then drop in a `.spec.yaml` + artefact pair.

There is also a no-op residual (`zero_residual`) you can use to sanity-check
the wiring without changing dynamics.

## How to enable them

1. Open [`configs/Scenario.yaml`](../configs/Scenario.yaml).
2. Set `active_scenario: hybrid_demo` (this scenario is provided and uses
   all three examples), **or** add a `hybrid:` block to one of your own
   scenarios — the schema is documented in [`docs/hybrid.md`](../docs/hybrid.md).
3. Run normally:

   ```bash
   python main.py
   ```

   The startup banner will report which hooks were wired in.

## Writing your own model

Pick the example that matches the plug point you need, copy the file,
keep the function signature the same, and replace the body with a call
to your model (sklearn, PyTorch, ONNX, lookup table, or anything callable).
Then point your scenario's `hybrid:` block at it via either:

- **dotted path** if your file is on `PYTHONPATH`:
  `"my_pkg.my_module:my_function"`
- **file path** if it's a one-off script:
  `"./my_models/my_model.py:my_function"`

That's the entire integration — load happens at simulator startup, the
callable runs inside the ODE solver, and the rest of the pipeline (plots,
CSV outputs, diagnostics) is unchanged.

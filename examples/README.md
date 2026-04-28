# Hybrid-mode examples

Each file here is a **drop-in replacement** for one part of the ADM1
model. Enable any combination from your scenario's `hybrid:` block in
[`../configs/Scenario.yaml`](../configs/Scenario.yaml) — no Python
changes required. Full guide: [`../docs/hybrid.md`](../docs/hybrid.md).

![Hybrid plug points](../docs/images/hybrid_plug_points.png)

## What each example shows

| File | Tier | Plug point | Pattern |
| ---- | ---- | ---- | ---- |
| [hybrid_rate_example.py](hybrid_rate_example.py) | 1 | `Rho_11` (acetoclastic methanogenesis) | classical Monod × Gaussian temperature factor — *"physics × learned correction"* |
| [hybrid_inhibition_example.py](hybrid_inhibition_example.py) | 1 | `I_nh3` (free-ammonia inhibition) | Hill sigmoid with tunable exponent — parametric replacement |
| [hybrid_residual_example.py](hybrid_residual_example.py) | 2 | residual on `dy/dt` | constant CH₄ bias (+ a `zero_residual` no-op for plumbing checks) — UDE-style additive correction |
| [hybrid_linear_regression_example.py](hybrid_linear_regression_example.py) | 1 | `Rho_2` (carbohydrate hydrolysis) | **real ML model**: linear regression fitted via `np.linalg.lstsq` on synthetic data — full *train offline → predict online* workflow in one file |

## Running them

Three demo scenarios in [`../configs/Scenario.yaml`](../configs/Scenario.yaml) wire these examples in:

| Scenario | What it activates |
| --- | --- |
| `hybrid_demo` | The first three examples together (rate + inhibition + residual hooks) |
| `hybrid_lr_demo` | The LR example loaded as a **raw callable** (the function in this directory) |
| `hybrid_lr_spec_demo` | The same trained LR loaded as a **HybridSpec sidecar** (from `../models/`) — same prediction, different load path |

To try one:

```yaml
# configs/Scenario.yaml
active_scenario: hybrid_demo     # or hybrid_lr_demo / hybrid_lr_spec_demo
```

```bash
python main.py
```

The startup banner reports which hooks were wired in.

---

## Writing your own hybrid component

You have **two options** for plugging your own model in. Pick whichever
fits your workflow.

### Option 1 — raw callable (fast iteration)

Best for prototyping, one-off experiments, or any model written in
Python. Just write a function with the right signature and point a YAML
at it.

1. Copy the example matching your plug point, keep the function signature.
2. Replace the body with your model (sklearn, PyTorch, ONNX, lookup, anything callable).
3. Reference it from your scenario:

   ```yaml
   hybrid:
     enabled: true
     rate_overrides:
       Rho_11: "my_pkg.my_module:my_function"        # dotted import path
       # or:
       Rho_11: "./local/my_model.py:my_function"     # plain file path
   ```

The simulator loads the function at startup and calls it inside the ODE solver.

### Option 2 — HybridSpec sidecar (saved, version-controlled models)

Best for trained models you want to ship as artefacts. You produce a
`.spec.yaml` describing the model + the artefact file (npz / joblib /
…), commit them under [`../models/`](../models/), and reference the
spec from a scenario. The simulator loads, validates, and inferences
for you.

```yaml
hybrid:
  enabled: true
  rate_overrides:
    Rho_2: "models/my_model.spec.yaml"              # auto-detects .spec.yaml
```

Save recipes for `linear_lstsq` (numpy) and `sklearn` (joblib) backends
are in [`../models/README.md`](../models/README.md). The integration
contract (required vs recommended fields, metadata schema) is in
[`../docs/hybrid.md`](../docs/hybrid.md#integration-contract).

---

## Validating the wiring

Use the `zero_residual` no-op from
[hybrid_residual_example.py](hybrid_residual_example.py) to sanity-check
the wiring without changing the dynamics:

```yaml
hybrid:
  enabled: true
  residual_correction: "examples.hybrid_residual_example:zero_residual"
```

With this enabled, the simulation must be **bit-for-bit identical** to a
pure ADM1 run on the same scenario. If it isn't, your wiring has a bug.

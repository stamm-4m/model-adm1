# `models/` — saved hybrid model artefacts

Trained hybrid components live here as **two-file pairs**:

```
models/
├── rho2.spec.yaml    ← declarative spec (target, inputs, model_type, metadata)
└── rho2.npz          ← model artefact (numpy / joblib / onnx / ...)
```

The `.spec.yaml` sidecar lets a saved model be loaded back later without
remembering what it was trained for.

> **Loader-only by design.** The simulator owns the spec format, the
> loader, and a few built-in backends. **It does not own training.** Use
> whatever ML stack you already know — sklearn, PyTorch, JAX, statsmodels,
> XGBoost, plain NumPy. As long as you produce a valid `.spec.yaml` +
> artefact pair, the integration works. The recipes below show the save
> step for each common backend.

## How to use a saved model

Point any scenario at the spec from `configs/Scenario.yaml`:

```yaml
hybrid:
  enabled: true
  rate_overrides:
    Rho_2: "models/rho2.spec.yaml"
```

The simulator detects the `.spec.yaml` suffix, reads the spec, loads the
artefact, and wires the resulting callable into the override slot. No
Python code changes required.

## Spec schema

```yaml
target: Rho_2                       # hook  (Rho_X | I_X | residual:<state>)
inputs: [X_ch, T_op, pH]            # feature names, in order — see docs/hybrid.md
model_type: linear_lstsq            # backend key  (linear_lstsq | sklearn | ...)
model_path: rho2.npz                # path (relative to spec or absolute)
metadata:                           # free-form, RECOMMENDED for traceability
  created_at: "2026-04-28T10:35:03+00:00"
  training_data: "synthetic"        # or path / hash / dataset name
  training_size: 2000
  notes: "..."
```

Built-in `target` forms:

- `Rho_X` — `X` in `1..19` → wires into `rate_overrides`
- `I_X` — one of `I_5..I_12, I_nh3` → wires into `inhibition_overrides`
- `residual:<state>` — `<state>` is any ADM1 state name → wires into `residual_correction`

## Recommended metadata fields

Not enforced by the loader, but the loader emits a `UserWarning` if any
of `created_at` or `training_data` is missing, since a model without
provenance is hard to audit a year later.

| Field | Why it matters |
| --- | --- |
| `created_at` | When was this trained? ISO-8601 timestamp recommended. |
| `training_data` | Where did the data come from? Path, dataset name, or content hash. |
| `training_size` | How many rows / samples? |
| `metric_*` | Validation metric(s) on a held-out set (e.g. `metric_mae`, `metric_r2`). |
| `adm1_version` | Git SHA of this repo at training time, so refits are reproducible. |
| `notes` | Free-form. Anything a colleague would want to know. |

Feel free to add domain-specific keys — the `metadata:` block is
intentionally schema-less.

---

## Save recipes

The recipes below produce `.spec.yaml` + artefact pairs that the loader
accepts. Use whichever backend matches the model you trained.

> The shared step is the spec sidecar. You can write the YAML by hand, or
> use the small `HybridSpec.save()` helper in `src/hybrid.py` — both
> produce identical files.

### `linear_lstsq` (NumPy, no extra deps)

Affine model `y = b0 + b1·x1 + ... + bn·xn`. The artefact is a `.npz`
file with a single key `coeffs` containing a 1-D float array of length
`1 + len(inputs)`: intercept first, then one weight per feature in
**the same order as `inputs`**.

```python
import numpy as np
from datetime import datetime, timezone
from src.hybrid import HybridSpec

# 1) Train however you like — here, OLS on a feature matrix Phi and target y.
#    Phi has shape (n_samples, 1 + len(inputs)) with a leading column of ones.
coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)   # shape (4,) for 3 features

# 2) Save the artefact (key MUST be "coeffs")
np.savez("models/my_rho.npz", coeffs=coeffs)

# 3) Save the spec sidecar
HybridSpec(
    target="Rho_2",
    inputs=["X_ch", "T_op", "pH"],
    model_type="linear_lstsq",
    model_path="my_rho.npz",        # relative to the spec file
    metadata={
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": "data/plant_2026Q1.parquet",
        "training_size": 12000,
        "metric_mae": 0.041,
        "metric_r2": 0.92,
    },
).save("models/my_rho.spec.yaml")
```

### `sklearn` (any estimator with `.predict`)

Anything with a scikit-learn-style `.predict([[...]])` method works:
`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`,
a full `Pipeline`, etc. The artefact is a joblib `.pkl`.

```python
import joblib
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from src.hybrid import HybridSpec

# 1) Train your estimator
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)         # X_train shape: (n, len(inputs))

# 2) Save the artefact
joblib.dump(model, "models/my_rho.joblib")

# 3) Save the spec sidecar
HybridSpec(
    target="Rho_2",
    inputs=["X_ch", "T_op", "pH"],
    model_type="sklearn",
    model_path="my_rho.joblib",
    metadata={
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": "data/plant_2026Q1.parquet",
        "training_size": len(X_train),
        "metric_mae": float(mae_on_val),
    },
).save("models/my_rho.spec.yaml")
```

> Requires `joblib` installed (or `scikit-learn`, which depends on it).

### Writing the YAML by hand

The `.spec.yaml` is just plain YAML — you can produce it from any
language or template engine:

```yaml
# models/my_rho.spec.yaml
target: Rho_2
inputs: [X_ch, T_op, pH]
model_type: linear_lstsq
model_path: my_rho.npz
metadata:
  created_at: "2026-04-28T10:00:00+00:00"
  training_data: "data/plant_2026Q1.parquet"
  training_size: 12000
```

## Adding a new backend

If your model doesn't fit `linear_lstsq` or `sklearn` (e.g. PyTorch,
ONNX, custom), add a backend in [`../src/hybrid.py`](../src/hybrid.py) by:

1. Adding a `_predict_<name>(artefact, x) -> float` function.
2. Adding a load branch in `_load_artefact` that returns the in-memory
   model.
3. Registering both in `_BACKEND_PREDICT`.

That's it — no other parts of the loader need to change.

## Should these files be committed?

- `*.spec.yaml` — **yes**, commit. Small, version-controlled metadata.
- Small `*.npz` artefacts — yes, fine to commit (a few KB).
- Large checkpoints (PyTorch, ONNX, > a few MB) — **no**, use Git LFS or
  fetch them at training time. Add a regeneration script and commit *that*.

## Regenerating the demo

The committed `rho2.spec.yaml` + `rho2.npz` were produced by training
the linear regression in
[`../examples/hybrid_linear_regression_example.py`](../examples/hybrid_linear_regression_example.py)
and saving with the recipe above. To reproduce:

```python
import numpy as np
from datetime import datetime, timezone
from src.hybrid import HybridSpec
from examples.hybrid_linear_regression_example import _fit_linear_regression

coeffs = _fit_linear_regression(n_samples=2000, seed=42)
np.savez("models/rho2.npz", coeffs=coeffs)
HybridSpec(
    target="Rho_2",
    inputs=["X_ch", "T_op", "pH"],
    model_type="linear_lstsq",
    model_path="rho2.npz",
    metadata={
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": "synthetic",
        "training_size": 2000,
        "ground_truth": "k_hyd_ch * X_ch * T_factor(T_op) * pH_factor(pH)",
        "coeffs_order": ["intercept", "X_ch", "T_op [K]", "pH"],
    },
).save("models/rho2.spec.yaml")
```

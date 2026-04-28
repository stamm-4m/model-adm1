# Hybrid Mode — Plug-and-Play ADM1 + ML

This simulator lets you mix the classical ADM1 with user-provided models
without modifying any code. Add a `hybrid:` block to a scenario in
[`configs/Scenario.yaml`](../configs/Scenario.yaml), point it at one or
more Python callables, and run as usual.

![Hybrid plug points](images/hybrid_plug_points.png)

| | Pure ADM1 | Hybrid mode |
| --- | --- | --- |
| What you change | nothing | one YAML block |
| What runs | classical Monod kinetics + ADM1 mass balances | classical kinetics, **with** any subset of: rate overrides, inhibition overrides, dy/dt residual |
| Overhead | none | none when no overrides are configured |

Three plug points are exposed:

| Tier | Plug point | What it lets you do |
| --- | --- | --- |
| 1 | `rate_overrides[Rho_X]` | replace one of the 19 classical process rates `Rho_1 .. Rho_19` |
| 1 | `inhibition_overrides[I_X]` | replace one of the inhibition factors `I_5..I_12, I_nh3` |
| 2 | `residual_correction` | add a UDE-style learned correction to `dy/dt` for any subset of the 38 states |

You can enable any combination — the simulator wires them in at startup,
runs them inside the ODE solver, and the rest of the pipeline (plots,
CSV outputs, diagnostics) is unchanged.

> **What this is not.** Tiers 1 + 2 use your callables at *inference time*.
> They do not enable end-to-end gradient-based training of a neural
> ODE — for that you would need to port the ODE right-hand side to a
> differentiable backend (JAX/`diffrax`, PyTorch/`torchdiffeq`).
> Most domain hybrid use cases (pre-trained NN slotted in, residual
> learning, lookup tables, regression) are well covered.

---

## Quick start

Two demo scenarios are provided out of the box:

- `hybrid_demo` — uses three hand-written example callables (rate, inhibition,
  residual) so you can see all plug points active at once.
- `hybrid_lr_demo` — uses a **real linear-regression model** (fitted on
  synthetic data with `np.linalg.lstsq` at module import) to replace
  `Rho_2` (carbohydrate hydrolysis). Closest to a real ML workflow.

Both run with the example callables in [`../examples/`](../examples/).

```bash
# In configs/Scenario.yaml
active_scenario: hybrid_demo
```

Run as usual:

```bash
python main.py
```

The startup banner reports which hooks were wired in:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ADM1 — Hybrid mode ENABLED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rate overrides       : Rho_11
  Inhibition overrides : I_nh3
  Residual correction  : enabled
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## YAML schema

Every key under `hybrid:` is optional. Omit a key (or remove the whole
`hybrid:` block, or set `enabled: false`) and the simulator runs pure
ADM1 — bit-for-bit identical to the no-hybrid case.

```yaml
scenarios:
  my_scenario:
    description: "..."
    initial_states: BSM2
    influent_mode: dynamic
    T_op: { value: 308.15, units: "K" }
    parameter_overrides: {}

    hybrid:
      enabled: true                       # required to activate

      # Tier 1 — replace process rates (any subset of Rho_1 .. Rho_19)
      rate_overrides:
        Rho_11: "examples.hybrid_rate_example:acetoclastic_rate_T_aware"
        # Rho_12: "my_pkg.my_module:hydrogenotrophic_rate"

      # Tier 1 — replace inhibition factors (any subset of I_5..I_12, I_nh3)
      inhibition_overrides:
        I_nh3: "examples.hybrid_inhibition_example:hill_nh3_inhibition"

      # Tier 2 — additive correction on dy/dt
      residual_correction: "examples.hybrid_residual_example:methane_bias_correction"
```

### Spec value forms

Each entry under `rate_overrides:` / `inhibition_overrides:` /
`residual_correction:` is a string. The simulator detects the form
automatically and dispatches to the right loader.

**Form 1 — raw callable** `"<location>:<function_name>"`:

- Dotted import path when your callable lives in a package on `PYTHONPATH`
  (the project root is on `PYTHONPATH` by default):

  ```yaml
  Rho_11: "examples.hybrid_rate_example:acetoclastic_rate_T_aware"
  ```

- File path when you have a one-off script and don't want to package it:

  ```yaml
  Rho_11: "./my_models/acetoclastic.py:predict"
  Rho_12: "/abs/path/to/model.py:predict"
  ```

**Form 2 — HybridSpec sidecar** `"path/to/<name>.spec.yaml"`:

```yaml
Rho_2: "models/rho2.spec.yaml"
```

This loads a saved, trained model artefact (no Python changes). The
sidecar describes what the artefact is and how to feed it; see the
**HybridSpec sidecars** section below.

---

## Callable signatures

### Tier 1 — rate override

```python
def my_rate(state: dict, inhib: dict, param) -> float:
    ...
```

| Argument | Type | What you receive |
| --- | --- | --- |
| `state` | `dict` | All 38 ADM1 state variables (`S_su`, `S_ac`, `X_ac`, …) plus the acid-base species (`S_H_ion`, `S_nh3`, `S_co2`, …). |
| `inhib` | `dict` | `{I_5, I_6, I_7, I_8, I_9, I_10, I_11, I_12, I_nh3}`, with any other inhibition overrides already applied. |
| `param` | `ADM1Parameters` | Read attributes like `param.k_m_ac`, `param.K_S_ac`, `param.T_op`. |

Returns a `float` — the rate value `Rho_X` in `kgCOD.m^-3.d^-1`.

See [examples/hybrid_rate_example.py](../examples/hybrid_rate_example.py).

### Tier 1 — inhibition override

```python
def my_inhibition(state: dict, param) -> float:
    ...
```

Returns a `float` in `[0, 1]` — `1.0` means no inhibition, `0.0` means full inhibition.

See [examples/hybrid_inhibition_example.py](../examples/hybrid_inhibition_example.py).

### Tier 2 — residual correction

```python
def my_residual(t: float, state: dict, param) -> dict:
    ...
```

Returns either:

- **dict form (recommended)** — `{state_name: dydt_delta}`. Only listed
  states are corrected; everything else is untouched.

  ```python
  return {"S_ch4": 0.001, "S_ac": -0.0005}
  ```

- **ndarray form** — a numpy array of length 38 aligned with `FULL_STATE_NAMES`.
  Useful for vectorised model outputs.

The returned values are **added** to the classical `dy/dt`. To leave a
state untouched, do not include it (or pass 0 in the array form).

See [examples/hybrid_residual_example.py](../examples/hybrid_residual_example.py).

---

## HybridSpec sidecars (saved model artefacts)

A `HybridSpec` is a tiny YAML sidecar that describes one trained model:
what hook it plugs into, what features it consumes, what backend it uses,
and where the artefact lives. It lets you ship a trained model as
**two version-controllable files** (a `.spec.yaml` + an artefact) and
swap models in YAML without touching Python.

### Schema

```yaml
target: Rho_2                       # hook (Rho_X | I_X | residual:<state>)
inputs: [X_ch, T_op, pH]            # feature names, in order
model_type: linear_lstsq            # backend key
model_path: rho2.npz                # artefact path (relative to spec or absolute)
metadata:                           # free-form, for traceability
  created_at: "2026-04-28T..."
  training_size: 2000
  notes: "..."
```

### How features are resolved

For each name in `inputs`, the loader extracts the value from the runtime
context:

| Name | Looked up in |
| --- | --- |
| any of the 38 `FULL_STATE_NAMES` (`S_su`, `X_ac`, ...) | `state[name]` |
| any of `I_5..I_12, I_nh3` | `inhib[name]` (rate overrides only) |
| `T_op` | `param.T_op` |
| `pH` | computed as `−log10(state["S_H_ion"])` |
| any other `ADM1Parameters` attribute | `getattr(param, name)` |

The order in `inputs` is the order the features are passed to the model,
so the saved coefficients/weights MUST match.

### Built-in backends

| `model_type` | Artefact format | Dependencies |
| --- | --- | --- |
| `linear_lstsq` | `.npz` with key `coeffs` (1-D: `[b0, b1, ..., b_n]`) | none — uses NumPy only |
| `sklearn` | `.joblib` (any sklearn estimator with `.predict([[…]])`) | `joblib` (or `scikit-learn`) |

Adding a new backend is a few lines in [`src/hybrid.py`](../src/hybrid.py):
register a `_predict_X` function in `_BACKEND_PREDICT` and a load branch
in `_load_artefact`.

### Wiring a saved model in

```yaml
# configs/Scenario.yaml
hybrid:
  enabled: true
  rate_overrides:
    Rho_2: "models/my_model.spec.yaml"
```

The two demo scenarios `hybrid_lr_demo` (raw callable) and
`hybrid_lr_spec_demo` (sidecar) wrap the same trained linear regression —
useful for an A/B comparison.

---

## Integration contract

The simulator is an **integration target**, not an ML framework. It owns
the spec format and the loader. **It does not own training.** You train
your model with whatever stack you already use; you produce a
`.spec.yaml` + artefact pair following the contract below; the simulator
does the rest.

### Required spec fields

| Field | Purpose |
| --- | --- |
| `target` | Which hook to plug into. Exactly one of: `Rho_X` (X in 1..19), an inhibition name in `{I_5..I_12, I_nh3}`, or `residual:<state>` where `<state>` is an ADM1 state name. |
| `inputs` | Ordered list of feature names. Each is resolved at inference from `state`, `inhib`, `param` (see *How features are resolved* above). The order is the order passed to your model — saved coefficients/weights must match. |
| `model_type` | Backend key. Built-in: `linear_lstsq`, `sklearn`. Adding new ones is a few lines in `src/hybrid.py`. |
| `model_path` | Path to the artefact file. Relative paths are resolved against the spec file's directory; absolute paths are used as-is. |

### Recommended `metadata` fields

`metadata` is intentionally schema-less — anything you put there is
preserved. The loader emits a `UserWarning` if any of `created_at` or
`training_data` is missing, since a model without provenance is hard to
audit later.

| Key | Why it matters |
| --- | --- |
| `created_at` | When was this trained? ISO-8601 timestamp recommended. |
| `training_data` | Where the data came from — path, dataset name, or content hash. |
| `training_size` | Number of training samples. |
| `metric_*` | Validation metrics on a held-out set (e.g. `metric_mae`, `metric_r2`). |
| `adm1_version` | Git SHA of this repo at training time, for reproducible refits. |
| `notes` | Free-form. Anything a colleague would want to know. |

### Save recipes

The full cookbook (with copy-paste recipes for `linear_lstsq` and
`sklearn`, plus how to write the YAML by hand) lives next to the
artefacts: see [`../models/README.md`](../models/README.md).

---

## What you can plug in

Any Python callable that matches the signature works. Common backends:

| Backend | How to wrap it |
| --- | --- |
| **scikit-learn** | `def predict(state, inhib, param): return float(model.predict([[state["S_ac"], state["X_ac"], ...]])[0])` |
| **PyTorch** | Wrap in `torch.no_grad()`, call `model(x).item()`. Load the model once at import time. |
| **ONNX Runtime** | Create the `InferenceSession` at import time, call `session.run(...)` in the function. |
| **Lookup table** | Read a CSV/parquet at import time, interpolate inside the function. |
| **Plain Python** | Often the simplest case — a hand-tuned formula like the Hill-style example. |

Heavy initialisation (loading a checkpoint, building a session) should
happen at module import time, **not** inside the called function — the
function runs at every solver step (potentially thousands of times per
simulated day). The
[hybrid_linear_regression_example.py](../examples/hybrid_linear_regression_example.py)
shows the full *train at import → infer in the loop* pattern with a
linear regression fitted by `np.linalg.lstsq`. The same skeleton works
unchanged for sklearn (replace `np.linalg.lstsq` with `model.fit`) or
PyTorch (load `state_dict` once, call `model(x).item()` inside the
override).

---

## Validating the hybrid run

A no-op residual is provided for plumbing checks:

```yaml
hybrid:
  enabled: true
  residual_correction: "examples.hybrid_residual_example:zero_residual"
```

With this enabled, the simulation result must be **bit-for-bit identical**
to a pure ADM1 run on the same scenario. If it isn't, your wiring has a
bug.

For a quick A/B comparison, run the same influent under two scenarios
(one classical, one with the hybrid block) and diff the
`results/dynamic_out.csv` files.

---

## FAQ

**Does enabling hybrid mode slow down the simulation?**
A small constant overhead per ODE step (one extra Python call per active
hook). For ML callables, total runtime is dominated by your model's
inference cost — pre-load weights at import time so you only pay that once.

**Can I override more than one rate or inhibition?**
Yes — list as many as you like under `rate_overrides:` /
`inhibition_overrides:`.

**Does my callable need to be pure / stateless?**
Stateless is recommended (and matches the no-hybrid behaviour). The
solver may call your function with non-monotonic times during step
rejection, so don't rely on time being monotonic.

**Can I read external data inside my callable?**
Yes, but cache it at import time. Re-reading from disk on every call is
the easiest way to make a simulation 100× slower.

**What if I typo a rate or inhibition name?**
The loader rejects unknown names at startup with a clear error
(e.g. "Unknown rate name 'Rho_111'") — typos surface immediately, not
mid-simulation.

"""
ADM1 hybrid-mode loader.

Wires registered hybrid models into an ADM1Reactor according to a scenario's

    hybrid:
      enabled: true
      use: [<model_name>, ...]

block. Each name must resolve to a model file under `models/<name>.yaml`,
loaded by `src.registry.load_registry()`.

Three plug points are supported, all optional:
  - Rate overrides       (Tier 1) : replace one of Rho_1 .. Rho_19
  - Inhibition overrides (Tier 1) : replace one of I_5..I_12, I_nh3
  - Residual correction  (Tier 2) : add a learned correction to dy/dt

Pure ADM1 = no `hybrid:` block, or `enabled: false`, or empty `use:`.

For the YAML schema and backend tags, see models/README.md.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np

from src.reactor import (
    FULL_STATE_NAMES,
    KNOWN_INHIBITION_NAMES,
    KNOWN_RATE_NAMES,
)


# ─────────────────────────────────────────────────────────────────────
# Feature extraction (state + inhib + param  →  feature vector)
# ─────────────────────────────────────────────────────────────────────


def _extract_features(input_names, state, inhib, param):
    """Build a 1-D float feature vector ordered as `input_names`."""
    values = []
    for name in input_names:
        if name == "pH":
            values.append(
                -float(np.log10(max(state.get("S_H_ion", 1e-14), 1e-14)))
            )
        elif name == "T_op":
            values.append(float(param.T_op))
        elif name in state:
            values.append(float(state[name]))
        elif inhib is not None and name in inhib:
            values.append(float(inhib[name]))
        elif hasattr(param, name):
            values.append(float(getattr(param, name)))
        else:
            raise KeyError(
                f"Unknown input feature '{name}'. Expected an ADM1 state name, "
                f"an inhibition name (rate overrides only), 'T_op', 'pH', or "
                f"an ADM1Parameters attribute."
            )
    return np.asarray(values, dtype=float)


# ─────────────────────────────────────────────────────────────────────
# Predict backends — small dispatch table; add new ones here.
# ─────────────────────────────────────────────────────────────────────


def _predict_linear_lstsq(model_artefact, x):
    """Affine model: y = b0 + dot(b[1:], x). Artefact is a 1-D coeff array."""
    coeffs = np.asarray(model_artefact, dtype=float)
    if coeffs.ndim != 1 or coeffs.shape[0] != 1 + x.shape[0]:
        raise ValueError(
            f"linear_lstsq expects coeffs shape (1+n_features,) = "
            f"({1 + x.shape[0]},), got {coeffs.shape}. "
            f"Check the model YAML's `inputs:` against the saved artefact."
        )
    return float(coeffs[0] + float(np.dot(coeffs[1:], x)))


def _predict_sklearn(model_artefact, x):
    """Wrap any sklearn-like estimator exposing .predict([[...]])."""
    return float(model_artefact.predict(x.reshape(1, -1))[0])


_BACKEND_PREDICT = {
    "linear_lstsq": _predict_linear_lstsq,
    "sklearn": _predict_sklearn,
}


def _load_artefact(backend: str, artefact_path: Path):
    """Load the model artefact from disk according to its `backend`."""
    path = Path(artefact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found: {path}")

    if backend == "linear_lstsq":
        data = np.load(path)
        if "coeffs" not in data:
            raise KeyError(
                f"Expected key 'coeffs' in {path}, got {list(data.keys())}."
            )
        return data["coeffs"]

    if backend == "sklearn":
        try:
            import joblib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "backend 'sklearn' requires joblib. Install it with "
                "`pip install joblib` (or scikit-learn)."
            ) from exc
        return joblib.load(path)

    raise ValueError(
        f"Unknown ML backend '{backend}'. Built-in: {sorted(_BACKEND_PREDICT)}."
    )


# ─────────────────────────────────────────────────────────────────────
# Raw-callable loader (backend: callable)
# ─────────────────────────────────────────────────────────────────────


def load_callable(spec: str) -> Callable:
    """
    Resolve a raw callable from a "location:function" string.

    Accepted forms:
      - "pkg.module:function"            (dotted import path)
      - "/abs/path/to/file.py:function"
      - "./relative/file.py:function"
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"Invalid callable spec '{spec}'. "
            f"Expected 'pkg.module:function' or '/path/to/file.py:function'."
        )

    location, func_name = spec.rsplit(":", 1)
    location = location.strip()
    func_name = func_name.strip()

    path = Path(location)
    is_file = path.suffix == ".py" or (path.exists() and path.is_file())

    if is_file:
        if not path.exists():
            raise FileNotFoundError(f"Hybrid callable file not found: {path}")
        module_name = f"_hybrid_{path.stem}"
        spec_obj = importlib.util.spec_from_file_location(module_name, path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load Python file: {path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(location)
        except ImportError as exc:
            raise ImportError(
                f"Could not import '{location}' for callable spec '{spec}'. "
                f"Make sure it is on PYTHONPATH or use a file-path spec."
            ) from exc

    if not hasattr(module, func_name):
        raise AttributeError(
            f"Module '{location}' has no attribute '{func_name}' (spec '{spec}')."
        )

    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"'{spec}' resolved to a non-callable ({type(fn).__name__}).")
    return fn


# ─────────────────────────────────────────────────────────────────────
# Compile a ModelEntry into a hook-shaped callable
# ─────────────────────────────────────────────────────────────────────


def _compile_entry(entry) -> Callable:
    """Turn a ModelEntry into a callable matching the right hook signature.

    Signatures by target:
      - "Rho_X"        : (state, inhib, param) -> float
      - "I_X"          : (state, param)        -> float
      - "residual:<S>" : (t, state, param)     -> dict[str, float]
    """
    target = entry.target

    if entry.backend == "callable":
        fn = load_callable(entry.artefact)
        fn.__name__ = f"hybrid_{target}_from_{entry.name}"
        return fn

    # ML-backed entry — resolve artefact relative to its source YAML.
    artefact_path = Path(entry.artefact)
    if not artefact_path.is_absolute():
        base = entry.source_path.parent if entry.source_path else Path(".")
        artefact_path = base / artefact_path
    artefact = _load_artefact(entry.backend, artefact_path)
    predict = _BACKEND_PREDICT[entry.backend]
    inputs = list(entry.inputs)

    if target in KNOWN_RATE_NAMES:
        def rate_override(state, inhib, param):
            x = _extract_features(inputs, state, inhib, param)
            return predict(artefact, x)
        rate_override.__name__ = f"hybrid_{target}_from_{entry.name}"
        return rate_override

    if target in KNOWN_INHIBITION_NAMES:
        def inhibition_override(state, param):
            x = _extract_features(inputs, state, None, param)
            return predict(artefact, x)
        inhibition_override.__name__ = f"hybrid_{target}_from_{entry.name}"
        return inhibition_override

    if isinstance(target, str) and target.startswith("residual:"):
        state_name = target.split(":", 1)[1]

        def residual_correction(t, state, param):
            x = _extract_features(inputs, state, None, param)
            return {state_name: predict(artefact, x)}

        residual_correction.__name__ = f"hybrid_residual_{state_name}_from_{entry.name}"
        return residual_correction

    # Defensive — registry validation should have caught this already.
    raise ValueError(f"Unknown target '{target}' on registered model '{entry.name}'.")


# ─────────────────────────────────────────────────────────────────────
# Wire hooks from a scenario's hybrid block into a reactor
# ─────────────────────────────────────────────────────────────────────


_LEGACY_KEYS = ("rate_overrides", "inhibition_overrides", "residual_correction")


def apply_hybrid_config(reactor, hybrid_cfg: dict, registry: dict) -> dict:
    """Wire hybrid hooks into `reactor` from a scenario's `hybrid:` block.

    Parameters
    ----------
    reactor    : ADM1Reactor instance
    hybrid_cfg : dict, the scenario's `hybrid:` block (may be empty / None).
                 Expected structure:

                   enabled: bool
                   use:     [<model_name>, ...]

    registry   : dict[name, ModelEntry], as returned by
                 src.registry.load_registry().

    Returns
    -------
    summary : dict for the startup banner. Keys:
        enabled, rate_overrides, inhibition_overrides, residual_correction
        (the three list values contain "<target> (<model_name>)" entries).

    Behaviour
    ---------
    - Empty / missing config, or `enabled: false` → silent no-op.
    - Unknown model name, duplicate target, or wrong key → raises clearly
      at startup.
    """
    summary = {
        "enabled": False,
        "rate_overrides": [],
        "inhibition_overrides": [],
        "residual_correction": [],
    }

    if not hybrid_cfg:
        return summary

    # Hard cutover: the old flat schema is gone.
    legacy_used = [k for k in _LEGACY_KEYS if k in hybrid_cfg]
    if legacy_used:
        raise ValueError(
            f"Scenario `hybrid:` block uses legacy key(s) {legacy_used}. "
            f"The schema is now:\n"
            f"    hybrid:\n"
            f"      enabled: true\n"
            f"      use: [<model_name>, ...]\n"
            f"and each <model_name> must be a file `models/<model_name>.yaml`. "
            f"Run `python main.py --list-models` to see what is registered "
            f"and `python main.py --list-hooks` for the hook catalog."
        )

    if not hybrid_cfg.get("enabled", False):
        return summary

    use_list = hybrid_cfg.get("use") or []
    if not isinstance(use_list, list):
        raise TypeError(
            f"`hybrid.use` must be a list of model names, "
            f"got {type(use_list).__name__}: {use_list!r}."
        )

    residual_owner = None  # which model already claimed the residual slot

    for model_name in use_list:
        if model_name not in registry:
            raise KeyError(
                f"Scenario references unknown hybrid model '{model_name}'. "
                f"Registered: {sorted(registry) or '(none)'}. "
                f"Add models/{model_name}.yaml or fix the scenario."
            )
        entry = registry[model_name]
        target = entry.target
        callable_fn = _compile_entry(entry)

        if target in KNOWN_RATE_NAMES:
            if target in reactor.rate_overrides:
                raise ValueError(
                    f"Two registered models target rate '{target}' in the same "
                    f"scenario. Use only one (got '{model_name}')."
                )
            reactor.rate_overrides[target] = callable_fn
            summary["rate_overrides"].append(f"{target} ({model_name})")
            continue

        if target in KNOWN_INHIBITION_NAMES:
            if target in reactor.inhibition_overrides:
                raise ValueError(
                    f"Two registered models target inhibition '{target}' in the "
                    f"same scenario. Use only one (got '{model_name}')."
                )
            reactor.inhibition_overrides[target] = callable_fn
            summary["inhibition_overrides"].append(f"{target} ({model_name})")
            continue

        if isinstance(target, str) and target.startswith("residual:"):
            if reactor.residual_correction is not None:
                raise ValueError(
                    f"Tier 2 supports one residual callable per scenario. "
                    f"'{model_name}' would replace the one already supplied by "
                    f"'{residual_owner}'. Compose multiple corrections in a "
                    f"single callable instead."
                )
            reactor.residual_correction = callable_fn
            residual_owner = model_name
            state_name = target.split(":", 1)[1]
            summary["residual_correction"].append(f"{state_name} ({model_name})")
            continue

        # Defensive — registry validation should have caught any other target.
        raise ValueError(
            f"Registered model '{model_name}' has unsupported target '{target}'."
        )

    summary["enabled"] = True
    return summary

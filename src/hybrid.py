"""
ADM1 hybrid-mode loader.

Wires user-provided callables (or saved model artefacts) into an
ADM1Reactor. Three plug points are supported, all optional:

  - Rate overrides       (Tier 1) : replace one of Rho_1 .. Rho_19
  - Inhibition overrides (Tier 1) : replace one of I_5..I_12, I_nh3
  - Residual correction  (Tier 2) : add a learned correction to dy/dt

Pure ADM1 = no hybrid block in YAML, or hybrid.enabled set to false.

YAML spec value forms
---------------------
Each entry under rate_overrides / inhibition_overrides / residual_correction
is a string that resolves to a callable. Two forms are accepted:

  1. Raw callable spec — points at a Python function:
        "examples.hybrid_rate_example:acetoclastic_rate_T_aware"
        "/abs/path/to/file.py:my_function"
        "./local/file.py:my_function"

  2. HybridSpec sidecar — points at a `.spec.yaml` file describing a
     trained, saved model:
        "models/rho2.spec.yaml"
        "/abs/path/to/some/model.spec.yaml"

The loader auto-detects the form from the value and dispatches to
`load_callable` or `load_hybrid` accordingly.

Expected callable signatures
----------------------------
rate_overrides[name](state: dict, inhib: dict, param) -> float
inhibition_overrides[name](state: dict, param) -> float
residual_correction(t: float, state: dict, param) -> dict | ndarray
"""

import importlib
import importlib.util
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List

import numpy as np
import yaml


# Metadata keys we encourage every saved model to carry. Their absence
# triggers a one-line UserWarning at load time — no failure, just a nudge.
RECOMMENDED_METADATA_KEYS = ("created_at", "training_data")

from src.reactor import (
    FULL_STATE_NAMES,
    KNOWN_INHIBITION_NAMES,
    KNOWN_RATE_NAMES,
)


# ─────────────────────────────────────────────────────────────────────
# HybridSpec — declarative description of one trained component
# ─────────────────────────────────────────────────────────────────────


@dataclass
class HybridSpec:
    """
    Declarative specification of one trained hybrid component.

    Persisted as a `.spec.yaml` sidecar next to the model artefact, so the
    artefact can be loaded later without remembering what it was trained for.

    Fields
    ------
    target : str
        Which ADM1 hook this component plugs into:
          - "Rho_X"          (X in 1..19)        → rate override
          - "I_X"            (I_5..I_12, I_nh3)  → inhibition override
          - "residual:<S>"   (<S> a state name)  → residual on dy/dt[S]
    inputs : list of str
        Names of the features the model consumes, resolved from
        (state, inhib, param) at inference time. Supported names:
          - any of the 38 FULL_STATE_NAMES         (taken from state)
          - any inhibition in KNOWN_INHIBITION_NAMES (taken from inhib;
            only available for rate overrides)
          - "T_op"                                 (taken from param.T_op)
          - "pH"                                   (= -log10(state["S_H_ion"]))
          - any other ADM1Parameters attribute    (taken from param)
    model_type : str
        Backend tag. Built-in: "linear_lstsq" | "sklearn".
    model_path : str
        Path to the artefact, relative to the spec file or absolute.
    metadata : dict
        Free-form dict for traceability (training size, MAE, R^2, date, ...).
    """

    target: str
    inputs: List[str]
    model_type: str
    model_path: str
    metadata: dict = field(default_factory=dict)

    def save(self, spec_path) -> None:
        """Write the spec to a YAML file (alongside its model artefact)."""
        spec_path = Path(spec_path)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, spec_path) -> "HybridSpec":
        """Load a spec from a YAML file."""
        with open(spec_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


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
                f"Unknown input feature '{name}'. Expected an ADM1 state "
                f"name, an inhibition name (rate overrides only), 'T_op', "
                f"'pH', or an ADM1Parameters attribute."
            )
    return np.asarray(values, dtype=float)


# ─────────────────────────────────────────────────────────────────────
# Predict backends
# ─────────────────────────────────────────────────────────────────────


def _predict_linear_lstsq(model_artefact, x):
    """Affine model: y = b0 + dot(b[1:], x). Artefact is a 1-D coeff array."""
    coeffs = np.asarray(model_artefact, dtype=float)
    if coeffs.ndim != 1 or coeffs.shape[0] != 1 + x.shape[0]:
        raise ValueError(
            f"linear_lstsq expects coeffs shape (1+n_features,) = "
            f"({1 + x.shape[0]},), got {coeffs.shape}. "
            f"Check spec.inputs vs the saved model."
        )
    return float(coeffs[0] + float(np.dot(coeffs[1:], x)))


def _predict_sklearn(model_artefact, x):
    """Wrap any sklearn-like estimator exposing .predict([[...]])."""
    return float(model_artefact.predict(x.reshape(1, -1))[0])


_BACKEND_PREDICT = {
    "linear_lstsq": _predict_linear_lstsq,
    "sklearn": _predict_sklearn,
}


def _load_artefact(model_type, model_path):
    """Load the model artefact from disk according to its `model_type`."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found: {path}")

    if model_type == "linear_lstsq":
        data = np.load(path)
        if "coeffs" not in data:
            raise KeyError(
                f"Expected key 'coeffs' in {path}, got {list(data.keys())}."
            )
        return data["coeffs"]

    if model_type == "sklearn":
        try:
            import joblib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "model_type 'sklearn' requires joblib. Install it with "
                "`pip install joblib` (or scikit-learn)."
            ) from exc
        return joblib.load(path)

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Built-in types: {sorted(_BACKEND_PREDICT)}."
    )


# ─────────────────────────────────────────────────────────────────────
# Sidecar loader: HybridSpec → callable matching the hook signature
# ─────────────────────────────────────────────────────────────────────


def load_hybrid(spec_path) -> Callable:
    """
    Load a HybridSpec + its model artefact, return a callable matching the
    override signature appropriate for `spec.target`.

    Parameters
    ----------
    spec_path : str or Path  — path to a `.spec.yaml` file.

    Returns
    -------
    callable : signature depends on spec.target:
        - "Rho_X"        : (state, inhib, param) -> float
        - "I_X"          : (state, param)        -> float
        - "residual:<S>" : (t, state, param)     -> dict[str, float]
    """
    spec_path = Path(spec_path)
    spec = HybridSpec.load(spec_path)

    # Soft provenance check — warn (don't fail) when the model has no
    # traceability info. See RECOMMENDED_METADATA_KEYS and the integration
    # contract in docs/hybrid.md.
    missing = [k for k in RECOMMENDED_METADATA_KEYS if not spec.metadata.get(k)]
    if missing:
        warnings.warn(
            f"HybridSpec at {spec_path} is missing recommended metadata "
            f"keys {missing}. The model will load, but provenance is unclear. "
            f"See docs/hybrid.md > 'Integration contract' for the recommended "
            f"metadata schema.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve model_path relative to the spec file (unless absolute)
    model_path = Path(spec.model_path)
    if not model_path.is_absolute():
        model_path = spec_path.parent / model_path

    artefact = _load_artefact(spec.model_type, model_path)
    predict = _BACKEND_PREDICT[spec.model_type]

    target = spec.target

    if target in KNOWN_RATE_NAMES:
        def rate_override(state, inhib, param):
            x = _extract_features(spec.inputs, state, inhib, param)
            return predict(artefact, x)
        rate_override.__name__ = f"hybrid_{target}_from_{spec_path.stem}"
        return rate_override

    if target in KNOWN_INHIBITION_NAMES:
        def inhibition_override(state, param):
            x = _extract_features(spec.inputs, state, None, param)
            return predict(artefact, x)
        inhibition_override.__name__ = f"hybrid_{target}_from_{spec_path.stem}"
        return inhibition_override

    if isinstance(target, str) and target.startswith("residual:"):
        state_name = target.split(":", 1)[1]
        if state_name not in FULL_STATE_NAMES:
            raise ValueError(
                f"residual target '{state_name}' is not a known ADM1 state. "
                f"Expected one of FULL_STATE_NAMES."
            )

        def residual_correction(t, state, param):
            x = _extract_features(spec.inputs, state, None, param)
            return {state_name: predict(artefact, x)}

        residual_correction.__name__ = (
            f"hybrid_residual_{state_name}_from_{spec_path.stem}"
        )
        return residual_correction

    raise ValueError(
        f"Unknown target '{target}' in {spec_path}. Expected one of:\n"
        f"  Rho_1 .. Rho_19\n"
        f"  {sorted(KNOWN_INHIBITION_NAMES)}\n"
        f"  residual:<state_name>"
    )


# ─────────────────────────────────────────────────────────────────────
# Raw-callable loader (unchanged, kept for backward compat)
# ─────────────────────────────────────────────────────────────────────


def load_callable(spec: str) -> Callable:
    """
    Resolve a raw callable from a "location:function" string.

    See module docstring for the supported `<location>` forms.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"Invalid callable spec '{spec}'. "
            f"Expected 'module.path:function' or '/path/to/file.py:function'."
        )

    location, func_name = spec.rsplit(":", 1)
    location = location.strip()
    func_name = func_name.strip()

    path = Path(location)
    is_file = path.suffix == ".py" or (path.exists() and path.is_file())

    if is_file:
        if not path.exists():
            raise FileNotFoundError(f"Hybrid model file not found: {path}")
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
                f"Could not import module '{location}' for hybrid hook '{spec}'. "
                f"Make sure it is on PYTHONPATH or use a file-path spec instead."
            ) from exc

    if not hasattr(module, func_name):
        raise AttributeError(
            f"Module '{location}' has no attribute '{func_name}' (hybrid spec '{spec}')."
        )

    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"'{spec}' resolved to a non-callable ({type(fn).__name__}).")
    return fn


# ─────────────────────────────────────────────────────────────────────
# Unified resolver: detect spec sidecars vs raw callables
# ─────────────────────────────────────────────────────────────────────


def _resolve_spec(value) -> Callable:
    """
    Resolve a YAML spec value into a callable.

    Accepted forms:
      - "module.path:function"          → load_callable
      - "/path/to/file.py:function"     → load_callable (file form)
      - "path/to/file.spec.yaml"        → load_hybrid (sidecar form)
      - "path/to/file.spec.yml"         → load_hybrid (sidecar form)
    """
    if not isinstance(value, str):
        raise TypeError(
            f"Hybrid spec value must be a string, got {type(value).__name__}: {value!r}"
        )

    s = value.strip()
    if s.endswith(".spec.yaml") or s.endswith(".spec.yml"):
        return load_hybrid(s)
    if ":" in s:
        return load_callable(s)
    raise ValueError(
        f"Invalid hybrid spec '{value}'. Expected either:\n"
        f"  'module.path:function' / '/path.py:function'  (raw callable)\n"
        f"  'path/to/file.spec.yaml'                       (HybridSpec sidecar)"
    )


# ─────────────────────────────────────────────────────────────────────
# Wire hooks from a config dict into a reactor
# ─────────────────────────────────────────────────────────────────────


def apply_hybrid_config(reactor, hybrid_cfg: dict) -> dict:
    """
    Wire hybrid hooks from a config dict into a reactor in place.

    Parameters
    ----------
    reactor     : ADM1Reactor instance
    hybrid_cfg  : dict, typically scenario["hybrid"] from Scenario.yaml.
                  Expected structure:

                    enabled: bool
                    rate_overrides:        { Rho_X: <spec>, ... }
                    inhibition_overrides:  { I_X:   <spec>, ... }
                    residual_correction:   <spec>

                  where <spec> is either a raw callable string
                  ("module:function" / "file.py:function") or a path to
                  a HybridSpec sidecar ("models/foo.spec.yaml").

    Returns
    -------
    summary : dict — useful for printing a startup banner. Keys:
        enabled, rate_overrides, inhibition_overrides, residual_correction.

    Behaviour
    ---------
    - Empty / missing config or enabled=false → silent no-op.
    - Unknown Rho_X / I_X names raise immediately (typo guard).
    - Each spec is resolved eagerly so import / load errors surface up-front.
    """
    summary = {
        "enabled": False,
        "rate_overrides": [],
        "inhibition_overrides": [],
        "residual_correction": False,
    }

    if not hybrid_cfg or not hybrid_cfg.get("enabled", False):
        return summary

    # --- Rate overrides (Tier 1) ---
    for rho_name, spec in (hybrid_cfg.get("rate_overrides") or {}).items():
        if rho_name not in KNOWN_RATE_NAMES:
            raise KeyError(
                f"Unknown rate name '{rho_name}' in hybrid.rate_overrides. "
                f"Expected one of Rho_1 .. Rho_19."
            )
        reactor.rate_overrides[rho_name] = _resolve_spec(spec)
        summary["rate_overrides"].append(rho_name)

    # --- Inhibition overrides (Tier 1) ---
    for inhib_name, spec in (hybrid_cfg.get("inhibition_overrides") or {}).items():
        if inhib_name not in KNOWN_INHIBITION_NAMES:
            raise KeyError(
                f"Unknown inhibition name '{inhib_name}' in hybrid.inhibition_overrides. "
                f"Expected one of {sorted(KNOWN_INHIBITION_NAMES)}."
            )
        reactor.inhibition_overrides[inhib_name] = _resolve_spec(spec)
        summary["inhibition_overrides"].append(inhib_name)

    # --- Residual correction (Tier 2) ---
    residual_spec = hybrid_cfg.get("residual_correction")
    if residual_spec:
        reactor.residual_correction = _resolve_spec(residual_spec)
        summary["residual_correction"] = True

    summary["enabled"] = True
    return summary

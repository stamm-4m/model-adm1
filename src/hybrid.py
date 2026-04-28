"""
ADM1 hybrid-mode loader.

Wires user-provided Python callables into an ADM1Reactor instance — loaded
from a YAML config block. Three plug points are supported, all of them
optional:

  - Rate overrides       (Tier 1) : replace one of Rho_1 .. Rho_19
  - Inhibition overrides (Tier 1) : replace one of I_5..I_12, I_nh3
  - Residual correction  (Tier 2) : add a learned correction to dy/dt

Pure ADM1 = no hybrid block in YAML, or hybrid.enabled set to false.

Callable specification format
-----------------------------
A callable is referenced by a string of the form

    "<location>:<function_name>"

where <location> is either:
  - an importable Python module (dotted path),
      e.g. "examples.hybrid_rate_example:acetoclastic_rate_T_aware"
  - a path to a .py file on disk,
      e.g. "/abs/path/to/my_model.py:predict"
            "./local/my_model.py:predict"

This lets you ship reusable hybrid components as a Python package OR drop
a one-off script next to a config without packaging anything.

Expected callable signatures
----------------------------
rate_overrides[name](state: dict, inhib: dict, param) -> float
inhibition_overrides[name](state: dict, param) -> float
residual_correction(t: float, state: dict, param) -> dict | ndarray
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Callable

from src.reactor import KNOWN_RATE_NAMES, KNOWN_INHIBITION_NAMES


def load_callable(spec: str) -> Callable:
    """
    Resolve a callable from a "location:function" string.

    See module docstring for the supported location forms.
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


def apply_hybrid_config(reactor, hybrid_cfg: dict) -> dict:
    """
    Wire hybrid hooks from a config dict into a reactor in place.

    Parameters
    ----------
    reactor     : ADM1Reactor instance
    hybrid_cfg  : dict, typically scenario["hybrid"] from Scenario.yaml.
                  Expected structure:

                    enabled: bool
                    rate_overrides:        { Rho_X: "module:func", ... }
                    inhibition_overrides:  { I_X:   "module:func", ... }
                    residual_correction:   "module:func"

    Returns
    -------
    summary : dict
        {
          "enabled": bool,
          "rate_overrides": [list of Rho names actually wired],
          "inhibition_overrides": [list of I names actually wired],
          "residual_correction": bool,
        }
        Useful for printing a startup banner.

    Behaviour
    ---------
    - Empty / missing config or enabled=false → silent no-op.
    - Unknown Rho_X / I_X names raise immediately (typo guard).
    - Each callable is loaded eagerly so import errors surface up-front.
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
        reactor.rate_overrides[rho_name] = load_callable(spec)
        summary["rate_overrides"].append(rho_name)

    # --- Inhibition overrides (Tier 1) ---
    for inhib_name, spec in (hybrid_cfg.get("inhibition_overrides") or {}).items():
        if inhib_name not in KNOWN_INHIBITION_NAMES:
            raise KeyError(
                f"Unknown inhibition name '{inhib_name}' in hybrid.inhibition_overrides. "
                f"Expected one of {sorted(KNOWN_INHIBITION_NAMES)}."
            )
        reactor.inhibition_overrides[inhib_name] = load_callable(spec)
        summary["inhibition_overrides"].append(inhib_name)

    # --- Residual correction (Tier 2) ---
    residual_spec = hybrid_cfg.get("residual_correction")
    if residual_spec:
        reactor.residual_correction = load_callable(residual_spec)
        summary["residual_correction"] = True

    summary["enabled"] = True
    return summary

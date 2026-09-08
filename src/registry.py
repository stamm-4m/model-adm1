"""
ADM1 hybrid-model registry.

The registry is just `models/*.yaml` — one YAML file per registered model.
The filename stem is the name scenarios use to reference it (e.g.
`models/rho2_lr_synthetic.yaml` is registered as `rho2_lr_synthetic`).

Each file declares one component that plugs into an ADM1 hook:

    description : str               (free-form, shown by --list-models)
    target      : str               which hook this model replaces:
                                      - "Rho_X"          (X in 1..19)
                                      - one of I_5..I_12, I_nh3
                                      - "residual:<state>"  (state in FULL_STATE_NAMES)
    backend     : str               "callable" | "linear_lstsq" | "sklearn"
    artefact    : str               for backend=callable: "pkg.module:fn"
                                                          or "path/to/file.py:fn"
                                    for ML backends:      path to the saved artefact
                                                          (relative to this YAML, or absolute)
    inputs      : [str]             ML backends only — ordered feature names
                                    resolved from state / inhib / param at inference
    metrics     : dict              free-form provenance (training_size, mae, created_at, ...)

This module loads + validates the registry. Compilation to hook-shaped
callables lives in `src.hybrid`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from src.reactor import (
    FULL_STATE_NAMES,
    KNOWN_INHIBITION_NAMES,
    KNOWN_RATE_NAMES,
)


DEFAULT_MODELS_DIR = Path("models")
SUPPORTED_BACKENDS = ("callable", "linear_lstsq", "sklearn")
REQUIRED_KEYS = ("target", "backend", "artefact")


@dataclass
class ModelEntry:
    """One row of the registry — fully resolved, ready to compile."""

    name: str
    description: str
    target: str
    backend: str
    artefact: str
    inputs: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    source_path: Optional[Path] = None  # the .yaml file this entry came from

    def short_target(self) -> str:
        """One-token form for tables: 'Rho_2', 'I_nh3', 'residual:S_ch4'."""
        return self.target


def _validate_target(target, source_path: Path) -> None:
    if target in KNOWN_RATE_NAMES:
        return
    if target in KNOWN_INHIBITION_NAMES:
        return
    if isinstance(target, str) and target.startswith("residual:"):
        state_name = target.split(":", 1)[1]
        if state_name in FULL_STATE_NAMES:
            return
        raise ValueError(
            f"{source_path}: residual target state '{state_name}' is not a "
            f"known ADM1 state. Run `python main.py --list-hooks` to see "
            f"the 38 valid state names."
        )
    raise ValueError(
        f"{source_path}: target '{target}' is not a valid hook. Expected one of:\n"
        f"  Rho_1 .. Rho_19\n"
        f"  {sorted(KNOWN_INHIBITION_NAMES)}\n"
        f"  residual:<state_name>\n"
        f"Run `python main.py --list-hooks` for the full catalog."
    )


def load_registry(models_dir=DEFAULT_MODELS_DIR) -> dict:
    """Glob `models_dir/*.yaml` and return {name: ModelEntry}.

    Raises on duplicate names, missing required keys, unknown backends, or
    invalid targets. Returns an empty dict if `models_dir` does not exist.
    """
    models_dir = Path(models_dir)
    registry: dict = {}
    if not models_dir.exists():
        return registry

    for path in sorted(models_dir.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: top-level must be a YAML mapping, "
                f"got {type(data).__name__}."
            )

        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"{path}: missing required keys {missing}.")

        backend = data["backend"]
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"{path}: backend '{backend}' is not supported. "
                f"Supported: {SUPPORTED_BACKENDS}."
            )

        _validate_target(data["target"], path)

        name = path.stem
        if name in registry:
            raise ValueError(f"Duplicate registered model name '{name}'.")

        registry[name] = ModelEntry(
            name=name,
            description=str(data.get("description", "")).strip(),
            target=data["target"],
            backend=backend,
            artefact=data["artefact"],
            inputs=list(data.get("inputs", [])),
            metrics=dict(data.get("metrics", {})),
            source_path=path,
        )

    return registry


# ─────────────────────────────────────────────────────────────────────
# Pretty-printers for --list-models / --list-hooks
# ─────────────────────────────────────────────────────────────────────


_RATE_DESCRIPTIONS = {
    "Rho_1":  "Disintegration of composites X_c",
    "Rho_2":  "Hydrolysis of carbohydrates X_ch",
    "Rho_3":  "Hydrolysis of proteins X_pr",
    "Rho_4":  "Hydrolysis of lipids X_li",
    "Rho_5":  "Uptake of sugars (S_su)",
    "Rho_6":  "Uptake of amino acids (S_aa)",
    "Rho_7":  "Uptake of LCFA (S_fa)",
    "Rho_8":  "Uptake of valerate (S_va)",
    "Rho_9":  "Uptake of butyrate (S_bu)",
    "Rho_10": "Uptake of propionate (S_pro)",
    "Rho_11": "Acetoclastic methanogenesis (S_ac -> CH4)",
    "Rho_12": "Hydrogenotrophic methanogenesis (H2 + CO2 -> CH4)",
    "Rho_13": "Decay of X_su",
    "Rho_14": "Decay of X_aa",
    "Rho_15": "Decay of X_fa",
    "Rho_16": "Decay of X_c4",
    "Rho_17": "Decay of X_pro",
    "Rho_18": "Decay of X_ac",
    "Rho_19": "Decay of X_h2",
}

_INHIBITION_DESCRIPTIONS = {
    "I_5":   "pH x N-limit on sugar uptake (Rho_5)",
    "I_6":   "pH x N-limit on amino-acid uptake (Rho_6)",
    "I_7":   "pH x N-limit x H2 on LCFA uptake (Rho_7)",
    "I_8":   "pH x N-limit x H2 on valerate (Rho_8)",
    "I_9":   "pH x N-limit x H2 on butyrate (Rho_9)",
    "I_10":  "pH x N-limit x H2 on propionate (Rho_10)",
    "I_11":  "pH x N-limit x NH3 on acetate (Rho_11)",
    "I_12":  "pH x N-limit on hydrogenotrophs (Rho_12)",
    "I_nh3": "Free-ammonia inhibition (component of I_11)",
}


def _inhib_sort_key(name: str):
    if name == "I_nh3":
        return (1, 0)
    return (0, int(name.split("_")[1]))


def format_registry(registry: dict) -> str:
    """Render the registry as a fixed-width table for --list-models."""
    if not registry:
        return (
            "No models registered.\n"
            "Add a YAML file under models/ to register one.\n"
            "Run `python main.py --list-hooks` to see what can be plugged into."
        )

    name_w = max(max(len(n) for n in registry), 4) + 2
    target_w = max(max(len(e.target) for e in registry.values()), 6) + 2
    backend_w = max(max(len(e.backend) for e in registry.values()), 7) + 2

    lines = [
        "Registered models  (models/*.yaml)",
        "-" * 78,
        f"  {'name'.ljust(name_w)}{'target'.ljust(target_w)}{'backend'.ljust(backend_w)}artefact",
        "-" * 78,
    ]
    for name in sorted(registry):
        e = registry[name]
        lines.append(
            f"  {name.ljust(name_w)}{e.target.ljust(target_w)}"
            f"{e.backend.ljust(backend_w)}{e.artefact}"
        )
        if e.description:
            indent = " " * (2 + name_w)
            lines.append(f"{indent}{e.description}")
    lines.append("-" * 78)
    lines.append(
        f"  {len(registry)} model(s) registered. "
        f"Use one from a scenario:  hybrid: {{ enabled: true, use: [<name>] }}"
    )
    return "\n".join(lines)


def format_hooks() -> str:
    """Render the full plug-point catalog for --list-hooks."""
    lines = [
        "ADM1 hook catalog",
        "=" * 78,
        "",
        "Tier 1 - Rate overrides   (set `target: Rho_X` in a model YAML)",
        "-" * 78,
    ]
    for i in range(1, 20):
        rho = f"Rho_{i}"
        lines.append(f"  {rho:<8}{_RATE_DESCRIPTIONS[rho]}")

    lines.append("")
    lines.append("Tier 1 - Inhibition overrides   (set `target: I_X`)")
    lines.append("-" * 78)
    for k in sorted(KNOWN_INHIBITION_NAMES, key=_inhib_sort_key):
        lines.append(f"  {k:<8}{_INHIBITION_DESCRIPTIONS[k]}")

    lines.append("")
    lines.append("Tier 2 - Residual corrections   (set `target: residual:<state>`)")
    lines.append("-" * 78)
    lines.append("  Any of the 38 ADM1 states (FULL_STATE_NAMES):")
    chunk_size = 6
    for i in range(0, len(FULL_STATE_NAMES), chunk_size):
        chunk = ", ".join(FULL_STATE_NAMES[i:i + chunk_size])
        lines.append(f"    {chunk}")

    lines.append("")
    lines.append(
        "Reference these targets from a YAML under models/, then list its "
        "name in `hybrid.use:`."
    )
    return "\n".join(lines)

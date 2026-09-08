"""
Render the README/docs diagrams as PNGs.

Run:
    python -m docs.scripts.render_diagrams

Outputs (overwritten on every run):
    docs/images/architecture.png
    docs/images/hybrid_plug_points.png
    docs/images/adm1_pipeline.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────
COLORS = {
    "input":   {"face": "#eef6fb", "edge": "#1a6e9e", "text": "#0b3a5b"},
    "core":    {"face": "#e6f7f2", "edge": "#117a65", "text": "#0b5345"},
    "output":  {"face": "#fff3e0", "edge": "#d45f30", "text": "#7a3210"},
    "hybrid":  {"face": "#f0eaff", "edge": "#6c3483", "text": "#4a235a"},
    "stage0":  {"face": "#eef0f3", "edge": "#566573", "text": "#1c2833"},
    "acido":   {"face": "#fff0e6", "edge": "#b9770e", "text": "#7f3b08"},
    "aceto":   {"face": "#f0eaff", "edge": "#6c3483", "text": "#4a235a"},
    "metha":   {"face": "#e6f7f2", "edge": "#117a65", "text": "#0b5345"},
}


def box(ax, x, y, w, h, text, kind="core", fontsize=10, weight="normal"):
    c = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=c["edge"], facecolor=c["face"],
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        color=c["text"], fontsize=fontsize, fontweight=weight,
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.6, style="->"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=18,
        linewidth=lw, color=color,
    ))


def label(ax, x, y, text, fontsize=9, color="#444", style="italic", ha="center"):
    ax.text(x, y, text, fontsize=fontsize, color=color, style=style, ha=ha, va="center")


def setup(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────
# 1. architecture.png — top-level data flow with optional hybrid layer
# ─────────────────────────────────────────────────────────────────────


def render_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=160)
    setup(ax, (0, 11.5), (0, 6))

    # Title
    ax.text(5.75, 5.65, "ADM1 simulator — runtime architecture",
            ha="center", fontsize=13, fontweight="bold", color="#222")

    # Inputs (left, three stacked) — vertical centres at 4.0 / 2.6 / 1.2
    box(ax, 0.2, 3.5, 2.1, 1.0, "Influent\n(CSV / YAML)", kind="input", fontsize=10)
    box(ax, 0.2, 2.1, 2.1, 1.0, "Scenario\n(YAML)", kind="input", fontsize=10)
    box(ax, 0.2, 0.7, 2.1, 1.0, "Parameters\n(YAML)", kind="input", fontsize=10)

    # Core simulator (centre)
    box(ax, 3.4, 0.7, 4.7, 3.8,
        "ADM1 reactor model\n\n"
        "38 ODEs · 19 process rates\nMonod kinetics + inhibitions\n\n"
        "scipy.solve_ivp (BDF)",
        kind="core", fontsize=10.5, weight="bold")

    # Optional hybrid lid — visually a layer ON TOP of the core
    box(ax, 3.4, 4.55, 4.7, 0.65,
        "Optional hybrid layer · rate / inhibition / residual hooks",
        kind="hybrid", fontsize=9.5, weight="bold")

    # Outputs (right)
    box(ax, 9.1, 3.4, 2.3, 1.1, "results/\n  dynamic_out.csv", kind="output", fontsize=9.5)
    box(ax, 9.1, 2.0, 2.3, 1.1, "results/figures/\n  *.png", kind="output", fontsize=9.5)
    box(ax, 9.1, 0.6, 2.3, 1.1, "console banner\n+ daily summary", kind="output", fontsize=9.5)

    # Arrows: inputs -> core
    arrow(ax, 2.3, 4.0, 3.4, 3.6)   # influent -> upper edge of core
    arrow(ax, 2.3, 2.6, 3.4, 2.6)   # scenario -> middle of core
    arrow(ax, 2.3, 1.2, 3.4, 1.6)   # parameters -> lower edge of core

    # Arrows: core -> outputs
    arrow(ax, 8.1, 3.6, 9.1, 3.95)
    arrow(ax, 8.1, 2.6, 9.1, 2.55)
    arrow(ax, 8.1, 1.6, 9.1, 1.15)

    fig.tight_layout()
    out = OUT_DIR / "architecture.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────
# 2. hybrid_plug_points.png — where the three hooks plug in
# ─────────────────────────────────────────────────────────────────────


def render_hybrid_plug_points():
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=160)
    setup(ax, (0, 11), (0, 5.5))

    ax.text(5.5, 5.1, "Hybrid plug points  ·  classical ADM1 with optional ML hooks",
            ha="center", fontsize=13, fontweight="bold", color="#222")

    # The classical pipeline blocks
    box(ax, 0.2, 2.4, 2.0, 1.0, "Inhibition\nfactors\nI_5..I_12, I_nh3", kind="acido", fontsize=9)
    box(ax, 2.7, 2.4, 2.0, 1.0, "Process rates\nRho_1..Rho_19", kind="aceto", fontsize=9)
    box(ax, 5.2, 2.4, 2.0, 1.0, "Mass balances\n(38 ODEs)", kind="metha", fontsize=9)
    box(ax, 7.7, 2.4, 2.0, 1.0, "ODE solver\nscipy.solve_ivp", kind="core", fontsize=9)

    arrow(ax, 2.2, 2.9, 2.7, 2.9)
    arrow(ax, 4.7, 2.9, 5.2, 2.9)
    arrow(ax, 7.2, 2.9, 7.7, 2.9)

    # Tier 1 — inhibition override
    box(ax, 0.2, 0.6, 2.0, 1.0,
        "Tier 1\nInhibition\noverride", kind="hybrid", fontsize=9, weight="bold")
    arrow(ax, 1.2, 1.6, 1.2, 2.4, color="#6c3483", lw=1.8)

    # Tier 1 — rate override
    box(ax, 2.7, 0.6, 2.0, 1.0,
        "Tier 1\nRate\noverride", kind="hybrid", fontsize=9, weight="bold")
    arrow(ax, 3.7, 1.6, 3.7, 2.4, color="#6c3483", lw=1.8)

    # Tier 2 — residual correction (after mass balances)
    box(ax, 5.2, 0.6, 2.0, 1.0,
        "Tier 2\nResidual\ncorrection", kind="hybrid", fontsize=9, weight="bold")
    arrow(ax, 6.2, 1.6, 6.2, 2.4, color="#6c3483", lw=1.8)

    # Bottom legend
    label(ax, 5.5, 0.15,
          "Each hook is a callable: classical when unset, your model when configured in YAML.",
          fontsize=9, color="#555", style="italic")

    fig.tight_layout()
    out = OUT_DIR / "hybrid_plug_points.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────
# 3. adm1_pipeline.png — biology / 4-stage pipeline
# ─────────────────────────────────────────────────────────────────────


def render_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.6), dpi=160)
    setup(ax, (0, 13), (0, 4.0))

    ax.text(6.5, 3.6, "ADM1 — 4-stage biochemical pipeline",
            ha="center", fontsize=13, fontweight="bold", color="#222")

    # Concise text — chemistry kept short to stay inside each box
    stages = [
        ("Disintegration", "complex matter\n→ carbs / prot / lipids", "stage0"),
        ("Hydrolysis",     "polymers\n→ monomers", "acido"),
        ("Acidogenesis",   "sugars · aa\n→ VFAs + H₂ + CO₂", "acido"),
        ("Acetogenesis",   "LCFA · VFAs\n→ acetate + H₂", "aceto"),
        ("Methanogenesis", "acetate / H₂+CO₂\n→ CH₄", "metha"),
    ]

    n = len(stages)
    box_w, box_h = 2.2, 1.9
    gap = (13 - n * box_w) / (n + 1)

    xs = []
    for i, (title, body, kind) in enumerate(stages):
        x = gap + i * (box_w + gap)
        c = COLORS[kind]
        # filled rounded box
        ax.add_patch(FancyBboxPatch(
            (x, 0.55), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6, edgecolor=c["edge"], facecolor=c["face"],
        ))
        # title (bold) + body (lighter)
        ax.text(x + box_w / 2, 0.55 + box_h * 0.72, title,
                ha="center", va="center",
                color=c["text"], fontsize=10.5, fontweight="bold")
        ax.text(x + box_w / 2, 0.55 + box_h * 0.32, body,
                ha="center", va="center",
                color=c["text"], fontsize=8.5)
        xs.append(x)

    arrow_y = 0.55 + box_h / 2
    for i in range(n - 1):
        x_from = xs[i] + box_w + 0.04
        x_to = xs[i + 1] - 0.04
        arrow(ax, x_from, arrow_y, x_to, arrow_y)

    fig.tight_layout()
    out = OUT_DIR / "adm1_pipeline.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print(f"Rendering diagrams to {OUT_DIR}/ ...")
    render_architecture()
    render_hybrid_plug_points()
    render_pipeline()
    print("Done.")


if __name__ == "__main__":
    main()

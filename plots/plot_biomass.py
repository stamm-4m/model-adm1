"""
Author: Margaux Bonal
Email: margaux.bonal@inrae.fr
Date: 04/2026

ADM1 — Plot: Microbial-biomass dynamics
========================================
Three panels in one figure:

  Top panel    : NORMALISED stacked areas (% of the total active biomass)
                 → shows the evolution of each population's relative share
                 → reveals dominance shifts without being hidden by absolute-
                   magnitude variations

  Middle panel : Stacked areas in ABSOLUTE values [kgCOD/m³]
                 → shows the total biomass and its quantitative evolution

  Bottom panel : Apparent net specific growth rate per population [d⁻¹]
                 computed as  Δ(ln X) / Δt   (= μ_app)
                 → early-warning signal: a fast-declining population shows
                   trouble before biogas drops.

The 7 ADM1 microbial populations:
  X_su  : sugar degraders                  (acidogenesis)
  X_aa  : amino-acid degraders             (acidogenesis)
  X_fa  : LCFA degraders                   (acetogenesis)
  X_c4  : valerate/butyrate degraders      (acetogenesis)
  X_pro : propionate degraders             (acetogenesis — often limiting)
  X_ac  : acetoclastic methanogens         (methanogenesis — ~70% CH4)
  X_h2  : hydrogenotrophic methanogens     (methanogenesis — ~30% CH4)

Usage:
    from plots.plot_biomass import plot_biomass
    plot_biomass(df, save_path="results/figures/biomass.png", show=True)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# ── Palette for the 7 populations ─────────────────────────────────────────────
# Organised by biochemical stage: warm → cool along the pipeline
POPULATIONS = {
    "X_su":  {"label": "$X_{su}$  sugars",            "color": "#e07b39", "step": "Acidogenesis"},
    "X_aa":  {"label": "$X_{aa}$  amino acids",       "color": "#c0392b", "step": "Acidogenesis"},
    "X_fa":  {"label": "$X_{fa}$  LCFA",              "color": "#8e44ad", "step": "Acetogenesis"},
    "X_c4":  {"label": "$X_{c4}$  C4 (val/but)",      "color": "#6c5ce7", "step": "Acetogenesis"},
    "X_pro": {"label": "$X_{pro}$  propionate",       "color": "#2980b9", "step": "Acetogenesis"},
    "X_ac":  {"label": "$X_{ac}$  acetate (methan.)", "color": "#27ae60", "step": "Methanogenesis"},
    "X_h2":  {"label": "$X_{h2}$  H₂ (methan.)",      "color": "#16a085", "step": "Methanogenesis"},
}

# Background colours per stage for annotations
STEP_COLORS = {
    "Acidogenesis":   "#fff0e6",
    "Acetogenesis":   "#f0eaff",
    "Methanogenesis": "#e6f7f2",
}


def _growth_rate(x_arr, t_arr):
    """
    Apparent net specific growth rate  Δ(ln X) / Δt  [d⁻¹].
    Returns an array of the same length (NaN at the first point).
    """
    dt   = np.diff(t_arr)
    dx   = np.diff(np.log(np.maximum(x_arr, 1e-12)))
    rate = np.where(dt > 0, dx / dt, np.nan)
    return np.concatenate([[np.nan], rate])


def plot_biomass(df, save_path: str = None, show: bool = True):
    """
    Parameters
    ----------
    df         : pandas DataFrame with ADM1 columns + a 'time' column
    save_path  : output path (PNG/PDF) — None = do not save
    show       : whether to show the matplotlib window
    """
    t    = df["time"].values
    keys = list(POPULATIONS.keys())

    # ── Data extraction ───────────────────────────────────────────────────────
    data = {k: df[k].values for k in keys}

    # Total active biomass
    X_tot = np.sum([data[k] for k in keys], axis=0)
    X_tot_safe = np.where(X_tot > 0, X_tot, np.nan)

    # Relative shares (%)
    pct = {k: 100 * data[k] / X_tot_safe for k in keys}

    # Net specific growth rate
    rates = {k: _growth_rate(data[k], t) for k in keys}

    # Stacked values for the absolute area plot
    stack_abs = np.array([data[k] for k in keys])
    stack_pct = np.array([pct[k]  for k in keys])

    colors = [POPULATIONS[k]["color"] for k in keys]
    labels = [POPULATIONS[k]["label"] for k in keys]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor("white")

    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[3, 2.5, 2],
                  hspace=0.10)

    ax_pct   = fig.add_subplot(gs[0])
    ax_abs   = fig.add_subplot(gs[1], sharex=ax_pct)
    ax_rate  = fig.add_subplot(gs[2], sharex=ax_pct)

    # ── Panel 1: normalised areas ─────────────────────────────────────────────
    ax_pct.stackplot(t, stack_pct, labels=labels, colors=colors, alpha=0.82)

    # Reference line: methanogen share (X_ac + X_h2)
    pct_meth = pct["X_ac"] + pct["X_h2"]
    ax_pct.plot(t, pct_meth, color="white", lw=1.2, ls="--", alpha=0.7,
                zorder=5)

    # Stage band annotations (background of the first panel)
    _add_step_bands(ax_pct, stack_pct, keys)

    ax_pct.set_ylim(0, 100)
    ax_pct.set_ylabel("Relative share  [%]", fontsize=10)
    ax_pct.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax_pct.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
    ax_pct.set_title(
        "Microbial population dynamics — ADM1", fontsize=12, pad=10
    )

    # Compact legend on the right
    _add_legend(ax_pct, keys, colors, labels)

    # Annotation for the methanogens dashed line
    ax_pct.text(
        t[len(t) // 10], float(np.nanmean(pct_meth)) + 2.5,
        "── total methanogens",
        color="white", fontsize=8, alpha=0.85,
    )

    plt.setp(ax_pct.get_xticklabels(), visible=False)

    # ── Panel 2: absolute areas ───────────────────────────────────────────────
    ax_abs.stackplot(t, stack_abs, colors=colors, alpha=0.78)

    # Total biomass curve (right axis)
    ax_abs_r = ax_abs.twinx()
    ax_abs_r.plot(t, X_tot, color="#2c3e50", lw=1.6, ls="-",
                  label="Total biomass")
    ax_abs_r.set_ylabel("Total  [kgCOD/m³]", fontsize=9,
                         rotation=-90, labelpad=14, color="#2c3e50")
    ax_abs_r.tick_params(axis="y", labelcolor="#2c3e50")
    ax_abs_r.spines[["top", "left"]].set_visible(False)

    ax_abs.set_ylabel("Biomass  [kgCOD/m³]", fontsize=10)
    ax_abs.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
    ax_abs.spines[["top", "right"]].set_visible(False)
    plt.setp(ax_abs.get_xticklabels(), visible=False)

    # ── Panel 3: net growth rate ──────────────────────────────────────────────
    ax_rate.axhline(0, color="#555", lw=0.8, ls="-", alpha=0.4)

    # Decline zone: pale red below 0
    ax_rate.axhspan(-999, 0, color="#fdecea", alpha=0.35, zorder=0)
    # Growth zone: pale green above 0
    ax_rate.axhspan(0, 999, color="#e8f5ee", alpha=0.35, zorder=0)

    for k in keys:
        r = rates[k]
        # Light smoothing (7-day moving average) for readability
        r_smooth = _moving_avg(r, window=7)
        ax_rate.plot(t, r_smooth,
                     color=POPULATIONS[k]["color"],
                     lw=1.2, alpha=0.85,
                     label=POPULATIONS[k]["label"])

    # Clip y-axis to avoid startup transients dominating the plot
    valid = []
    for k in keys:
        v = rates[k][~np.isnan(rates[k])]
        if len(v):
            valid.extend(v.tolist())
    if valid:
        p5, p95 = np.percentile(valid, 5), np.percentile(valid, 95)
        margin = max(abs(p5), abs(p95)) * 1.5
        ax_rate.set_ylim(-margin, margin)

    ax_rate.set_ylabel("Net growth rate  [d⁻¹]", fontsize=10)
    ax_rate.set_xlabel("Time  [d]", fontsize=10)
    ax_rate.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
    ax_rate.spines[["top", "right"]].set_visible(False)

    # Zone annotations
    ax_rate.text(t[3], ax_rate.get_ylim()[1] * 0.85,
                 "growth", color="#27ae60", fontsize=8, alpha=0.8)
    ax_rate.text(t[3], ax_rate.get_ylim()[0] * 0.85,
                 "decline", color="#c0392b", fontsize=8, alpha=0.8)

    # ── Y-label alignment ─────────────────────────────────────────────────────
    fig.align_ylabels([ax_pct, ax_abs, ax_rate])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Biomass plot saved → {save_path}")

    if show:
        plt.show()

    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _add_step_bands(ax, stack_pct, keys):
    """
    Add stage labels to the right edge of the normalised panel, placed at the
    centre of the corresponding colour band.
    Acidogenesis = X_su + X_aa  |  Acetogenesis = X_fa + X_c4 + X_pro
    Methanogenesis = X_ac + X_h2
    """
    step_groups = {
        "Acidogenesis":   ["X_su", "X_aa"],
        "Acetogenesis":   ["X_fa", "X_c4", "X_pro"],
        "Methanogenesis": ["X_ac", "X_h2"],
    }
    step_colors_text = {
        "Acidogenesis":   "#7f3b08",
        "Acetogenesis":   "#4a235a",
        "Methanogenesis": "#0b5345",
    }

    # Compute cumulative heights to place labels at t_mid
    t_mid_idx = len(stack_pct[0]) // 2
    cumul = 0.0
    for step_name, step_keys in step_groups.items():
        idx_list = [list(POPULATIONS.keys()).index(k) for k in step_keys]
        band_height = sum(stack_pct[i][t_mid_idx] for i in idx_list)
        y_center = cumul + band_height / 2
        cumul += band_height

        if band_height > 3:   # only show the label if the band is visible
            ax.text(
                ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 1,
                y_center,
                step_name,
                ha="right", va="center",
                fontsize=8, color=step_colors_text[step_name],
                fontweight="bold", alpha=0.85,
            )


def _add_legend(ax, keys, colors, labels):
    """Compact legend placed outside the plot area."""
    patches = [
        mpatches.Patch(color=colors[i], label=labels[i], alpha=0.82)
        for i in range(len(keys))
    ]
    ax.legend(
        handles=patches,
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        framealpha=0.85,
        title="Populations",
        title_fontsize=9,
    )


def _moving_avg(arr, window=7):
    """Simple moving average — handles NaNs and edges."""
    result = np.full_like(arr, np.nan, dtype=float)
    valid  = ~np.isnan(arr)
    if not valid.any():
        return result
    # Replace NaNs with 0 for the convolution
    filled = np.where(valid, arr, 0.0)
    kernel = np.ones(window) / window
    smooth = np.convolve(filled, kernel, mode="same")
    # Edge correction (fewer valid points near the boundaries)
    count  = np.convolve(valid.astype(float), kernel * window, mode="same")
    count  = np.where(count > 0, count, np.nan)
    smooth = smooth / (count / window)
    result[valid] = smooth[valid]
    return result
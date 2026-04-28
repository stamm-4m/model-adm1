"""
Author: Margaux Bonal
Email: margaux.bonal@inrae.fr
Date: 04/2026

ADM1 — Plot: pH, alkalinity and digester stability
====================================================
Four panels in one figure:

  Panel 1 — pH
    · pH curve computed from S_H_ion: pH = -log10(S_H_ion)
    · Green zone : optimal range for acetoclastic methanogens (6.8 – 7.6)
    · Orange zone: stress range                                (6.5 – 6.8  and  7.6 – 8.0)
    · Red zone   : critical (acidification / alkalinisation)
    · Reference lines pH_LL_ac and pH_UL_ac read from `param`
    · Dynamic annotation: time spent outside the optimal zone

  Panel 2 — Total alkalinity and its components
    · S_IC       : total inorganic carbon       [kmol C/m³]  — left axis
    · S_hco3_ion : bicarbonate HCO₃⁻            [M]          — left axis
    · S_co2      : dissolved CO₂                [M]          — left axis (dashed)
    · HCO₃⁻ / IC ratio                          [%]          — right axis
    → HCO₃⁻ is the main buffer. If its share of IC drops (CO₂ rises),
      the digester is losing its buffering capacity.

  Panel 3 — VFA / alkalinity ratio  (stability indicator)
    · Total VFA = S_va_ion + S_bu_ion + S_pro_ion + S_ac_ion  [mol/L]
    · Alkalinity proxy = S_hco3_ion                            [M]
    · VFA/Alk ratio  [-]
    · Zones: < 0.3 stable | 0.3–0.6 caution | > 0.6 acidification risk
    → This is the best early operational indicator:
      the ratio rises several days before pH drops.

  Panel 4 — Inorganic nitrogen and free ammonia
    · S_IN total              [kmol N/m³]  — left axis
    · S_nh3 (free NH₃)        [M]          — left axis, thick line
    · S_nh4_ion (NH₄⁺)        [M]          — left axis, dashed
    · K_I_nh3 threshold        [M]          — red dashed line
    → Free NH₃ is the inhibitory form (not NH₄⁺). Its concentration
      depends on BOTH pH and temperature — hence the coupling.

Usage:
    from plots.plot_pH_alkalinity import plot_pH_alkalinity
    plot_pH_alkalinity(df, param, save_path="results/figures/pH_alkalinity.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# ── Palette ───────────────────────────────────────────────────────────────────
C_PH      = "#1a5276"   # navy blue     — pH
C_IC      = "#117a65"   # forest green  — total inorganic carbon
C_HCO3    = "#27ae60"   # green         — bicarbonate
C_CO2     = "#d4ac0d"   # gold          — dissolved CO2
C_RATIO_C = "#0e6655"   # dark green    — HCO3/IC ratio
C_VFA     = "#c0392b"   # red           — total VFAs
C_ALK     = "#2980b9"   # blue          — alkalinity proxy
C_RATIO_V = "#922b21"   # dark red      — VFA/Alk ratio
C_NH3     = "#7d3c98"   # purple        — free ammonia
C_NH4     = "#a569bd"   # light purple  — ammonium
C_IN      = "#6c3483"   # dark purple   — total inorganic nitrogen

# pH-zone colours
Z_OPTIMAL = "#eafaf1"   # very pale green
Z_STRESS  = "#fef9e7"   # very pale yellow
Z_CRIT    = "#fdedec"   # very pale red

# VFA/Alk-zone colours
Z_STABLE  = "#eafaf1"
Z_WARN    = "#fef9e7"
Z_RISK    = "#fdedec"


def plot_pH_alkalinity(df, param, save_path: str = None, show: bool = True):
    """
    Parameters
    ----------
    df         : pandas DataFrame with ADM1 columns + a 'time' column
    param      : ADM1Parameters object (for pH_LL_ac, pH_UL_ac, K_I_nh3)
    save_path  : output path — None = do not save
    show       : whether to show the matplotlib window
    """
    t = df["time"].values

    # ── Derived quantities ────────────────────────────────────────────────────

    # pH
    S_H_ion = df["S_H_ion"].values
    pH      = -np.log10(np.maximum(S_H_ion, 1e-14))

    # Alkalinity
    S_IC      = df["S_IC"].values          # kmol C/m³
    S_hco3    = df["S_hco3_ion"].values    # M
    S_co2_d   = df["S_co2"].values         # M  (dissolved CO2)

    # HCO3/IC ratio — IC in kmol C/m³, HCO3 in M (= kmol/m³) → directly comparable
    IC_safe     = np.where(S_IC > 0, S_IC, np.nan)
    ratio_hco3  = 100 * S_hco3 / IC_safe   # %

    # Total VFAs (ionic forms, in M)
    VFA_total = (df["S_va_ion"].values
                 + df["S_bu_ion"].values
                 + df["S_pro_ion"].values
                 + df["S_ac_ion"].values)

    # VFA / alkalinity ratio (alkalinity proxy = HCO3)
    alk_safe  = np.where(S_hco3 > 1e-6, S_hco3, np.nan)
    ratio_vfa = VFA_total / alk_safe

    # Nitrogen
    S_IN      = df["S_IN"].values       # kmol N/m³
    S_nh3     = df["S_nh3"].values      # M
    S_nh4     = df["S_nh4_ion"].values  # M

    # Reference parameters
    pH_LL = getattr(param, "pH_LL_ac", 6.0)
    pH_UL = getattr(param, "pH_UL_ac", 7.0)
    K_I_nh3 = getattr(param, "K_I_nh3", 0.0018)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 13))
    fig.patch.set_facecolor("white")

    gs = GridSpec(4, 1, figure=fig,
                  height_ratios=[2.5, 2, 2, 2],
                  hspace=0.10)

    ax_ph   = fig.add_subplot(gs[0])
    ax_alk  = fig.add_subplot(gs[1], sharex=ax_ph)
    ax_vfa  = fig.add_subplot(gs[2], sharex=ax_ph)
    ax_n    = fig.add_subplot(gs[3], sharex=ax_ph)

    ax_alk_r = ax_alk.twinx()
    ax_vfa_r = ax_vfa.twinx()
    ax_n_r   = ax_n.twinx()

    t_max = t[-1]

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — pH
    # ═══════════════════════════════════════════════════════════════════════════

    # Coloured zones
    ax_ph.axhspan(6.8, 7.6, color=Z_OPTIMAL, zorder=0, label="Optimal zone (6.8–7.6)")
    ax_ph.axhspan(6.5, 6.8, color=Z_STRESS,  zorder=0, alpha=0.7)
    ax_ph.axhspan(7.6, 8.0, color=Z_STRESS,  zorder=0, alpha=0.7)
    ax_ph.axhspan(4.0, 6.5, color=Z_CRIT,    zorder=0, alpha=0.5)
    ax_ph.axhspan(8.0, 9.5, color=Z_CRIT,    zorder=0, alpha=0.5)

    # Model reference lines
    ax_ph.axhline(pH_LL, color=C_PH, lw=0.9, ls="--", alpha=0.5)
    ax_ph.axhline(pH_UL, color=C_PH, lw=0.9, ls="--", alpha=0.5)
    ax_ph.text(t_max * 0.01, pH_LL - 0.12,
               f"pH_LL_ac = {pH_LL:.1f}", fontsize=8, color=C_PH, alpha=0.7)
    ax_ph.text(t_max * 0.01, pH_UL + 0.06,
               f"pH_UL_ac = {pH_UL:.1f}", fontsize=8, color=C_PH, alpha=0.7)

    # pH curve
    ax_ph.plot(t, pH, color=C_PH, lw=2.0, zorder=5, label="pH")

    # Annotation: time outside the optimal zone
    out_of_zone = np.sum((pH < 6.8) | (pH > 7.6))
    pct_out     = 100 * out_of_zone / max(len(pH), 1)
    ax_ph.text(0.98, 0.05,
               f"{pct_out:.1f} % of time outside optimal zone",
               transform=ax_ph.transAxes,
               ha="right", va="bottom", fontsize=8.5,
               color="#c0392b" if pct_out > 10 else "#117a65",
               bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="white", edgecolor="none", alpha=0.8))

    # pH statistics
    pH_mean = np.nanmean(pH)
    pH_min  = np.nanmin(pH)
    pH_max  = np.nanmax(pH)
    ax_ph.axhline(pH_mean, color=C_PH, lw=0.7, ls=":", alpha=0.6)
    ax_ph.text(t_max * 0.98, pH_mean + 0.04,
               f"mean {pH_mean:.2f}", fontsize=8, color=C_PH,
               ha="right", alpha=0.8)

    _style_ax(ax_ph,
              ylabel="pH  [—]",
              ylim=(max(4.0, pH_min - 0.5), min(9.5, pH_max + 0.5)),
              hide_x=True)

    # Panel-1 legend
    handles_ph = [
        mpatches.Patch(color=Z_OPTIMAL, label="Optimal zone  (6.8–7.6)"),
        mpatches.Patch(color=Z_STRESS,  label="Stress zone   (6.5–6.8 / 7.6–8.0)"),
        mpatches.Patch(color=Z_CRIT,    label="Critical zone"),
        plt.Line2D([0],[0], color=C_PH, lw=2, label="Simulated pH"),
    ]
    ax_ph.legend(handles=handles_ph, fontsize=8.5, loc="upper right",
                 framealpha=0.85, ncol=2)
    ax_ph.set_title("pH, alkalinity and digester stability — ADM1",
                    fontsize=12, pad=10)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Alkalinity
    # ═══════════════════════════════════════════════════════════════════════════

    # Total IC area (background)
    ax_alk.fill_between(t, S_IC, alpha=0.12, color=C_IC)
    l_ic,   = ax_alk.plot(t, S_IC,    color=C_IC,   lw=1.5,
                          label="$S_{IC}$ total inorganic carbon  [kmol C/m³]")
    l_hco3, = ax_alk.plot(t, S_hco3,  color=C_HCO3, lw=2.0,
                          label="$S_{HCO_3^-}$ bicarbonate  [M]")
    l_co2,  = ax_alk.plot(t, S_co2_d, color=C_CO2,  lw=1.2, ls="--",
                          label="$S_{CO_2}$ dissolved CO₂  [M]")

    # HCO3/IC ratio on right axis
    ax_alk_r.plot(t, ratio_hco3, color=C_RATIO_C, lw=1.2, ls="-.",
                  alpha=0.8, label="HCO₃⁻ / IC  [%]")
    ax_alk_r.axhline(80, color=C_RATIO_C, lw=0.7, ls=":", alpha=0.5)
    ax_alk_r.text(t_max * 0.01, 81,
                  "80 % (effective buffering threshold)",
                  fontsize=7.5, color=C_RATIO_C, alpha=0.75)
    ax_alk_r.set_ylabel("HCO₃⁻ / IC  [%]", fontsize=9,
                         rotation=-90, labelpad=14, color=C_RATIO_C)
    ax_alk_r.tick_params(axis="y", labelcolor=C_RATIO_C)
    ax_alk_r.set_ylim(0, 110)
    ax_alk_r.spines[["top", "left"]].set_visible(False)

    handles_alk = [l_ic, l_hco3, l_co2,
                   plt.Line2D([0],[0], color=C_RATIO_C, lw=1.2, ls="-.",
                              label="HCO₃⁻ / IC  [%]")]
    ax_alk.legend(handles=handles_alk, fontsize=8.5, loc="upper right",
                  framealpha=0.85)
    _style_ax(ax_alk, ylabel="Concentration  [kmol/m³ or M]", hide_x=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — VFA / alkalinity ratio
    # ═══════════════════════════════════════════════════════════════════════════

    # Stability zones
    ax_vfa_r.axhspan(0,    0.3,  color=Z_STABLE, alpha=0.55, zorder=0)
    ax_vfa_r.axhspan(0.3,  0.6,  color=Z_WARN,   alpha=0.55, zorder=0)
    ax_vfa_r.axhspan(0.6,  5.0,  color=Z_RISK,   alpha=0.40, zorder=0)

    # Total VFAs and alkalinity on the left axis
    ax_vfa.fill_between(t, VFA_total, alpha=0.15, color=C_VFA)
    l_vfa, = ax_vfa.plot(t, VFA_total, color=C_VFA, lw=1.8,
                         label="Total VFAs  (va+bu+pro+ac)  [M]")
    l_alk, = ax_vfa.plot(t, S_hco3,    color=C_ALK, lw=1.4, ls="--",
                         label="HCO₃⁻ (alkalinity)  [M]")

    # Ratio on the right axis
    l_ratio, = ax_vfa_r.plot(t, ratio_vfa, color=C_RATIO_V, lw=2.0, zorder=5,
                              label="VFA/Alk ratio  [—]")
    ax_vfa_r.axhline(0.3, color=C_RATIO_V, lw=0.8, ls=":", alpha=0.7)
    ax_vfa_r.axhline(0.6, color=C_RATIO_V, lw=0.8, ls=":", alpha=0.7)
    ax_vfa_r.text(t_max * 0.98, 0.31, "0.3  stable",
                  ha="right", fontsize=7.5, color=C_RATIO_V, alpha=0.8)
    ax_vfa_r.text(t_max * 0.98, 0.61, "0.6  risk",
                  ha="right", fontsize=7.5, color=C_RATIO_V, alpha=0.8)
    ax_vfa_r.set_ylabel("VFA / alkalinity ratio  [—]", fontsize=9,
                         rotation=-90, labelpad=14, color=C_RATIO_V)
    ax_vfa_r.tick_params(axis="y", labelcolor=C_RATIO_V)
    ax_vfa_r.set_ylim(0, max(1.5, np.nanpercentile(ratio_vfa, 98) * 1.3))
    ax_vfa_r.spines[["top", "left"]].set_visible(False)

    handles_vfa = [l_vfa, l_alk,
                   plt.Line2D([0],[0], color=C_RATIO_V, lw=2,
                              label="VFA / Alk ratio  [—]"),
                   mpatches.Patch(color=Z_STABLE, label="Stable    (< 0.3)"),
                   mpatches.Patch(color=Z_WARN,   label="Caution   (0.3–0.6)"),
                   mpatches.Patch(color=Z_RISK,   label="Risk      (> 0.6)")]
    ax_vfa.legend(handles=handles_vfa, fontsize=8.5, loc="upper right",
                  framealpha=0.85, ncol=2)
    _style_ax(ax_vfa, ylabel="Concentration  [M]", hide_x=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 4 — Inorganic nitrogen and free NH3
    # ═══════════════════════════════════════════════════════════════════════════

    # S_IN on the left axis (kmol/m³)
    ax_n.fill_between(t, S_IN, alpha=0.10, color=C_IN)
    l_in, = ax_n.plot(t, S_IN, color=C_IN, lw=1.4, ls="--",
                      label="$S_{IN}$ total inorganic nitrogen  [kmol N/m³]")

    # NH3 and NH4 on the right axis (M)
    l_nh4, = ax_n_r.plot(t, S_nh4, color=C_NH4, lw=1.2, ls="--", alpha=0.7,
                          label="$S_{NH_4^+}$ ammonium  [M]")
    l_nh3, = ax_n_r.plot(t, S_nh3, color=C_NH3, lw=2.2,
                          label="$S_{NH_3}$ free ammonia  [M]")

    # Inhibition threshold K_I_nh3
    ax_n_r.axhline(K_I_nh3, color="#c0392b", lw=1.0, ls="--", alpha=0.85)
    ax_n_r.text(t_max * 0.01, K_I_nh3 * 1.06,
                f"$K_{{I,NH_3}}$ = {K_I_nh3:.4f} M  (inhibition threshold)",
                fontsize=8, color="#c0392b", alpha=0.9)

    # Annotation: % of time NH3 exceeds the threshold
    over_threshold = np.sum(S_nh3 > K_I_nh3)
    pct_over = 100 * over_threshold / max(len(S_nh3), 1)
    if pct_over > 0:
        ax_n_r.text(0.98, 0.92,
                    f"NH₃ > threshold: {pct_over:.1f} % of time",
                    transform=ax_n_r.transAxes,
                    ha="right", va="top", fontsize=8.5,
                    color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="none", alpha=0.8))

    ax_n_r.set_ylabel("$NH_3$ / $NH_4^+$  [M]", fontsize=9,
                       rotation=-90, labelpad=14, color=C_NH3)
    ax_n_r.tick_params(axis="y", labelcolor=C_NH3)
    ax_n_r.spines[["top", "left"]].set_visible(False)

    handles_n = [l_in, l_nh3, l_nh4,
                 plt.Line2D([0],[0], color="#c0392b", lw=1.0, ls="--",
                            label=f"Threshold $K_{{I,NH_3}}$ = {K_I_nh3:.4f} M")]
    ax_n.legend(handles=handles_n, fontsize=8.5, loc="upper right",
                framealpha=0.85)
    _style_ax(ax_n, ylabel="$S_{IN}$  [kmol N/m³]",
              xlabel="Time  [d]", hide_x=False)

    # ── Final touches ─────────────────────────────────────────────────────────
    fig.align_ylabels([ax_ph, ax_alk, ax_vfa, ax_n])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  pH/alkalinity plot saved → {save_path}")

    if show:
        plt.show()

    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax, ylabel="", xlabel="", ylim=None, hide_x=False):
    """Apply a consistent style to an axis."""
    ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylim:
        ax.set_ylim(ylim)
    if hide_x:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
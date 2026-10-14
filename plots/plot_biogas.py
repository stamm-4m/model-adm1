"""
Author: Margaux Bonal
Email: margaux.bonal@inrae.fr
Date: 04/2026

ADM1 — Plot: Biogas production
==============================
Single figure with two panels:
  - Top panel   : biogas flow q_gas (wet, T_op, P_gas) and the dry-normal biogas / CH4 flows
                  [m³/d, + mL/d scale]  — left axis
                  H₂ partial pressure     [bar]   — right axis (trace)
  - Bottom panel: CH₄ partial pressure    [bar]   — left axis (filled area)
                  CO₂ partial pressure    [bar]   — left axis (filled area)
                  CH₄ share in biogas     [%]     — right axis (dashed line)

CH₄ % is computed as:
    p_total_gas = p_CH4 + p_CO2 + p_H2   (water vapour ignored — not tracked)
    pct_CH4 = 100 * p_CH4 / p_total_gas

Standalone usage:
    from plots.plot_biogas import plot_biogas
    plot_biogas(df, param, show=True)

From main.py:
    plot_biogas(df, param)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.reactor import ADM1Reactor


# ── Consistent palette ────────────────────────────────────────────────────────
C_QGAS   = "#1a6e9e"   # navy blue      — biogas flow
C_CH4    = "#2a9d5c"   # green          — methane
C_CO2    = "#d45f30"   # brick orange   — CO2
C_H2     = "#9b59b6"   # purple         — hydrogen (trace, right axis)
C_PCT    = "#555555"   # dark grey      — % CH4
C_ZONE   = "#e8f5ee"   # very pale green — optimal CH4 % zone
C_WARN   = "#fff3cd"   # pale yellow    — warning zone (pH / composition)


def _compute_q_gas(df, reactor):
    """
    Reconstruct gas partial pressures and gas flow from ADM1 gas states.

    Convention used:
    - S_gas_h2  : gas-phase H2 state   [kgCOD.m^-3]
    - S_gas_ch4 : gas-phase CH4 state  [kgCOD.m^-3]
    - S_gas_co2 : gas-phase CO2 state  [kmolC.m^-3]
    """
    param = reactor.param
    p_h2, p_ch4, p_co2 = reactor.gas_state_to_partial_pressures(
        S_gas_h2=df["S_gas_h2"].values,
        S_gas_ch4=df["S_gas_ch4"].values,
        S_gas_co2=df["S_gas_co2"].values,
    )

    # same q_gas law as the simulation (gas_law_patm); the P_atm law needs S_h2/S_ch4/S_co2
    q_gas = np.asarray(reactor.compute_gas_outputs(df)["q_gas"], dtype=float)

    return q_gas, p_ch4, p_co2, p_h2


def plot_biogas(df, param, save_path: str = None, show: bool = True):
    """
    Parameters
    ----------
    df         : pandas DataFrame with ADM1 columns + a 'time' column
    param      : ADM1Parameters object
    save_path  : output path (PNG/PDF) — None = do not save
    show       : whether to show the matplotlib window
    """
    t = df["time"].values
    reactor = ADM1Reactor(param, constants=None)

    # ── Data ──────────────────────────────────────────────────────────────────
    q_gas, p_ch4, p_co2, p_h2 = _compute_q_gas(df, reactor)
    gas_out = reactor.compute_gas_outputs(df)
    q_gas_norm_dry = np.asarray(gas_out["q_gas_norm_dry"], dtype=float)   # dry, 0 C, 1.01325 bar
    q_ch4_norm_dry = np.asarray(gas_out["q_ch4_norm_dry"], dtype=float)

    p_total_gas = p_ch4 + p_co2 + p_h2                  # H2O excluded
    pct_ch4 = np.where(p_total_gas > 0,
                       100 * p_ch4 / p_total_gas,
                       np.nan)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("white")

    # Two panels, height ratio 2:3
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 3], hspace=0.08)
    ax_top    = fig.add_subplot(gs[0])
    ax_bot    = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top_r  = ax_top.twinx()   # right axis, top panel    → p_H2
    ax_bot_r  = ax_bot.twinx()   # right axis, bottom panel → % CH4

    # ── Top panel: biogas flow + p_H2 ─────────────────────────────────────────
    ax_top.fill_between(t, q_gas, alpha=0.18, color=C_QGAS)
    l1, = ax_top.plot(t, q_gas, color=C_QGAS, lw=1.8,
                      label="Biogas flow  $q_{gas}$  (wet, T_op, P_gas)")
    l1n, = ax_top.plot(t, q_gas_norm_dry, color=C_QGAS, lw=1.1, ls=":",
                       label="Biogas  (dry, normal: 0 °C, 1 atm)")
    l1c, = ax_top.plot(t, q_ch4_norm_dry, color=C_CH4, lw=1.3,
                       label="CH₄  (dry, normal)")

    ax_top_r.plot(t, p_h2 * 1000, color=C_H2, lw=1.0, ls="--", alpha=0.75,
                  label="$p_{H_2}$  [mbar]")

    ax_top.set_ylabel("Gas flow  [m³/d]", color=C_QGAS, fontsize=10)
    ax_top.tick_params(axis="y", labelcolor=C_QGAS)
    ax_top.set_ylim(bottom=0)
    # Same axis in mL/d (linear unit change): normal-dry curves read directly in NmL/d,
    # which is what lab gas counters report.
    ax_top_ml = ax_top.secondary_yaxis(-0.13, functions=(lambda x: x * 1e6, lambda x: x / 1e6))
    ax_top_ml.set_ylabel("[mL/d]  (NmL/d for dry-normal curves)", fontsize=8.5, color="#555555")
    ax_top_ml.tick_params(axis="y", labelsize=8, labelcolor="#555555")
    ax_top_r.set_ylabel("$p_{H_2}$  [mbar]", color=C_H2, fontsize=9,
                         rotation=-90, labelpad=14)
    ax_top_r.tick_params(axis="y", labelcolor=C_H2)
    ax_top_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Top-panel legend
    handles_top = [l1, l1n, l1c,
                   Line2D([0],[0], color=C_H2, lw=1.0, ls="--",
                          label="$p_{H_2}$  [mbar]  (right axis)")]
    ax_top.legend(handles=handles_top, fontsize=9, loc="upper left",
                  framealpha=0.7)
    ax_top.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax_top.set_title("Biogas production — ADM1", fontsize=12, pad=10)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Bottom panel: biogas composition + % CH4 ──────────────────────────────

    # Optimal CH4 composition zone: 55–70 %
    ax_bot_r.axhspan(55, 70, color=C_ZONE, alpha=0.55, zorder=0,
                     label="Optimal CH₄ zone  (55–70 %)")
    # Alert zone: < 50 %
    ax_bot_r.axhspan(0, 50, color=C_WARN, alpha=0.35, zorder=0,
                     label="Alert zone  (< 50 %)")

    # Filled partial-pressure areas
    ax_bot.stackplot(t, p_ch4, p_co2,
                     labels=["$p_{CH_4}$  [bar]", "$p_{CO_2}$  [bar]"],
                     colors=[C_CH4, C_CO2], alpha=0.55)

    # Sharp contour lines on top of the areas
    ax_bot.plot(t, p_ch4,        color=C_CH4, lw=1.4, alpha=0.9)
    ax_bot.plot(t, p_ch4 + p_co2, color=C_CO2, lw=1.0, alpha=0.7, ls="--")

    # % CH4 on right axis
    l_pct, = ax_bot_r.plot(t, pct_ch4, color=C_PCT, lw=1.8, ls="-.",
                            label="CH₄ in biogas  [%]", zorder=5)

    # Reference line at 65 % (typical BSM2 value)
    ax_bot_r.axhline(65, color=C_PCT, lw=0.8, ls=":", alpha=0.6)
    ax_bot_r.text(0.01, 65.8, "65 % (BSM2 ref.)", transform=ax_bot_r.get_yaxis_transform(),
                  color=C_PCT, fontsize=8, alpha=0.8)

    ax_bot.set_ylabel("Partial pressure  [bar]", fontsize=10)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlabel("Time  [d]", fontsize=10)
    ax_bot_r.set_ylabel("Composition  [% CH₄]", fontsize=10,
                         rotation=-90, labelpad=16)
    ax_bot_r.set_ylim(0, 100)
    ax_bot_r.tick_params(axis="y", labelcolor=C_PCT)
    ax_bot.grid(True, ls=":", lw=0.6, alpha=0.6)

    # Bottom-panel legend — built explicitly (each entry exactly once)
    handles_bot = [
        Patch(facecolor=C_CH4, alpha=0.55, label="$p_{CH_4}$  [bar]"),
        Patch(facecolor=C_CO2, alpha=0.55, label="$p_{CO_2}$  [bar]"),
        Line2D([0], [0], color=C_PCT, lw=1.8, ls="-.", label="CH₄ in dry biogas  [%]"),
        Patch(facecolor=C_ZONE, alpha=0.55, label="Optimal CH₄ zone  (55–70 %)"),
        Patch(facecolor=C_WARN, alpha=0.35, label="Alert zone  (< 50 %)"),
    ]
    ax_bot.legend(handles=handles_bot, fontsize=9, loc="lower right", framealpha=0.75, ncol=2)

    # ── Dynamic annotations ───────────────────────────────────────────────────
    # Max biogas flow
    idx_max = int(np.argmax(q_gas))
    ax_top.annotate(
        f"max {q_gas[idx_max]:.4g} m³/d",
        xy=(t[idx_max], q_gas[idx_max]),
        xytext=(12, -12), textcoords="offset points",
        fontsize=8, color=C_QGAS,
        arrowprops=dict(arrowstyle="->", color=C_QGAS, lw=0.8),
    )

    # Mean % CH4 (NaNs ignored)
    pct_mean = np.nanmean(pct_ch4)
    ax_bot_r.axhline(pct_mean, color=C_PCT, lw=0.6, ls="--", alpha=0.5)
    ax_bot_r.text(0.01, pct_mean + 1.2, f"mean {pct_mean:.1f} %",
                  transform=ax_bot_r.get_yaxis_transform(),
                  color=C_PCT, fontsize=8, alpha=0.8)

    # ── Final touches ─────────────────────────────────────────────────────────
    for ax in [ax_top, ax_bot]:
        ax.spines[["top", "right"]].set_visible(False)
    for ax in [ax_top_r, ax_bot_r]:
        ax.spines[["top", "left"]].set_visible(False)

    ax_bot.set_xlim(t[0], t[-1])        # shared x: no blank margin before t_start / after t_end
    fig.align_ylabels([ax_top, ax_bot])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Biogas plot saved → {save_path}")

    if show:
        plt.show()

    return fig

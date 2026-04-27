"""
Autor :  Margaux Bonal 
Email : margaux.bonal@inrae.fr
Date : 04/2026

ADM1 — Graphique : pH, alcalinité et stabilité du digesteur
=============================================================
Quatre panneaux sur une même figure :

  Panneau 1 — pH
    · Courbe pH calculée depuis S_H_ion : pH = -log10(S_H_ion)
    · Zone verte  : pH optimal méthanogènes acétoclastes   (6.8 – 7.6)
    · Zone orange : zone de stress                          (6.5 – 6.8  et  7.6 – 8.0)
    · Zone rouge  : zone critique (acidification/alcalinisation)
    · Lignes de référence pH_LL_ac et pH_UL_ac lues depuis param
    · Annotation dynamique : durée passée hors zone optimale

  Panneau 2 — Alcalinité totale et ses composantes
    · S_IC   : carbone inorganique total       [kmol C/m³]  — axe gauche
    · S_hco3_ion : bicarbonate HCO₃⁻           [M]          — axe gauche
    · S_co2  : CO₂ dissous                     [M]          — axe gauche (tirets)
    · Ratio HCO₃⁻/IC                          [%]          — axe droit
    → Le HCO₃⁻ est le tampon principal. Si sa part dans l'IC chute
      (CO₂ augmente), le digesteur perd sa capacité tampon.

  Panneau 3 — Ratio VFA / alcalinité  (indicateur de stabilité)
    · VFA totaux = S_va_ion + S_bu_ion + S_pro_ion + S_ac_ion  [mol/L]
    · Alcalinité proxy = S_hco3_ion                             [M]
    · Ratio VFA/Alk  [-]
    · Zones : < 0.3 stable | 0.3–0.6 attention | > 0.6 risque d'acidification
    → C'est le meilleur indicateur opérationnel précoce :
      le ratio monte plusieurs jours avant que le pH ne chute.

  Panneau 4 — Azote inorganique et ammoniac libre
    · S_IN total              [kmol N/m³]  — axe gauche
    · S_nh3 (NH₃ libre)       [M]          — axe gauche, ligne épaisse
    · S_nh4_ion (NH₄⁺)        [M]          — axe gauche, tirets
    · Seuil K_I_nh3            [M]          — ligne rouge pointillée
    → NH₃ libre est la forme inhibitrice (pas NH₄⁺). Sa concentration
      dépend du pH ET de la température, d'où l'importance du couplage.

Utilisation :
    from plots.plot_pH_alkalinity import plot_pH_alkalinity
    plot_pH_alkalinity(df, param, save_path="results/figures/pH_alkalinity.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# ── Palette ───────────────────────────────────────────────────────────────────
C_PH      = "#1a5276"   # bleu marine   — pH
C_IC      = "#117a65"   # vert sapin    — carbone inorganique total
C_HCO3    = "#27ae60"   # vert          — bicarbonate
C_CO2     = "#d4ac0d"   # or            — CO2 dissous
C_RATIO_C = "#0e6655"   # vert foncé    — ratio HCO3/IC
C_VFA     = "#c0392b"   # rouge         — VFA totaux
C_ALK     = "#2980b9"   # bleu          — alcalinité proxy
C_RATIO_V = "#922b21"   # rouge foncé   — ratio VFA/Alk
C_NH3     = "#7d3c98"   # violet        — ammoniac libre
C_NH4     = "#a569bd"   # violet clair  — ammonium
C_IN      = "#6c3483"   # violet foncé  — azote inorganique total

# Couleurs zones pH
Z_OPTIMAL = "#eafaf1"   # vert très pâle
Z_STRESS  = "#fef9e7"   # jaune très pâle
Z_CRIT    = "#fdedec"   # rouge très pâle

# Couleurs zones VFA/Alk
Z_STABLE  = "#eafaf1"
Z_WARN    = "#fef9e7"
Z_RISK    = "#fdedec"


def plot_pH_alkalinity(df, param, save_path: str = None, show: bool = True):
    """
    Paramètres
    ----------
    df         : DataFrame pandas avec colonnes ADM1 + colonne 'time'
    param      : objet ADM1Parameters (pour pH_LL_ac, pH_UL_ac, K_I_nh3)
    save_path  : chemin de sauvegarde — None = pas de sauvegarde
    show       : afficher la fenêtre matplotlib
    """
    t = df["time"].values

    # ── Calculs dérivés ───────────────────────────────────────────────────────

    # pH
    S_H_ion = df["S_H_ion"].values
    pH      = -np.log10(np.maximum(S_H_ion, 1e-14))

    # Alcalinité
    S_IC      = df["S_IC"].values          # kmol C/m³
    S_hco3    = df["S_hco3_ion"].values    # M
    S_co2_d   = df["S_co2"].values         # M  (CO2 dissous)

    # Ratio HCO3/IC  — IC est en kmol C/m³, HCO3 en M (= kmol/m³) → directement comparables
    IC_safe     = np.where(S_IC > 0, S_IC, np.nan)
    ratio_hco3  = 100 * S_hco3 / IC_safe   # %

    # VFA totaux (formes ioniques, en M)
    VFA_total = (df["S_va_ion"].values
                 + df["S_bu_ion"].values
                 + df["S_pro_ion"].values
                 + df["S_ac_ion"].values)

    # Ratio VFA / alcalinité  (proxy alcalinité = HCO3)
    alk_safe  = np.where(S_hco3 > 1e-6, S_hco3, np.nan)
    ratio_vfa = VFA_total / alk_safe

    # Azote
    S_IN      = df["S_IN"].values       # kmol N/m³
    S_nh3     = df["S_nh3"].values      # M
    S_nh4     = df["S_nh4_ion"].values  # M

    # Paramètres de référence
    pH_LL = getattr(param, "pH_LL_ac", 6.0)
    pH_UL = getattr(param, "pH_UL_ac", 7.0)
    K_I_nh3 = getattr(param, "K_I_nh3", 0.0018)

    # ── Mise en page ──────────────────────────────────────────────────────────
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
    # PANNEAU 1 — pH
    # ═══════════════════════════════════════════════════════════════════════════

    # Zones colorées
    ax_ph.axhspan(6.8, 7.6, color=Z_OPTIMAL, zorder=0, label="Zone optimale (6.8–7.6)")
    ax_ph.axhspan(6.5, 6.8, color=Z_STRESS,  zorder=0, alpha=0.7)
    ax_ph.axhspan(7.6, 8.0, color=Z_STRESS,  zorder=0, alpha=0.7)
    ax_ph.axhspan(4.0, 6.5, color=Z_CRIT,    zorder=0, alpha=0.5)
    ax_ph.axhspan(8.0, 9.5, color=Z_CRIT,    zorder=0, alpha=0.5)

    # Lignes de référence du modèle
    ax_ph.axhline(pH_LL, color=C_PH, lw=0.9, ls="--", alpha=0.5)
    ax_ph.axhline(pH_UL, color=C_PH, lw=0.9, ls="--", alpha=0.5)
    ax_ph.text(t_max * 0.01, pH_LL - 0.12,
               f"pH_LL_ac = {pH_LL:.1f}", fontsize=8, color=C_PH, alpha=0.7)
    ax_ph.text(t_max * 0.01, pH_UL + 0.06,
               f"pH_UL_ac = {pH_UL:.1f}", fontsize=8, color=C_PH, alpha=0.7)

    # Courbe pH
    ax_ph.plot(t, pH, color=C_PH, lw=2.0, zorder=5, label="pH")

    # Annotation : durée hors zone optimale
    hors_zone = np.sum((pH < 6.8) | (pH > 7.6))
    pct_hors  = 100 * hors_zone / max(len(pH), 1)
    ax_ph.text(0.98, 0.05,
               f"{pct_hors:.1f} % du temps hors zone optimale",
               transform=ax_ph.transAxes,
               ha="right", va="bottom", fontsize=8.5,
               color="#c0392b" if pct_hors > 10 else "#117a65",
               bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="white", edgecolor="none", alpha=0.8))

    # Statistiques pH
    pH_mean = np.nanmean(pH)
    pH_min  = np.nanmin(pH)
    pH_max  = np.nanmax(pH)
    ax_ph.axhline(pH_mean, color=C_PH, lw=0.7, ls=":", alpha=0.6)
    ax_ph.text(t_max * 0.98, pH_mean + 0.04,
               f"moy. {pH_mean:.2f}", fontsize=8, color=C_PH,
               ha="right", alpha=0.8)

    _style_ax(ax_ph,
              ylabel="pH  [—]",
              ylim=(max(4.0, pH_min - 0.5), min(9.5, pH_max + 0.5)),
              hide_x=True)

    # Légende panneau 1
    handles_ph = [
        mpatches.Patch(color=Z_OPTIMAL, label="Zone optimale  (6.8–7.6)"),
        mpatches.Patch(color=Z_STRESS,  label="Zone de stress  (6.5–6.8 / 7.6–8.0)"),
        mpatches.Patch(color=Z_CRIT,    label="Zone critique"),
        plt.Line2D([0],[0], color=C_PH, lw=2, label="pH simulé"),
    ]
    ax_ph.legend(handles=handles_ph, fontsize=8.5, loc="upper right",
                 framealpha=0.85, ncol=2)
    ax_ph.set_title("pH, alcalinité et stabilité du digesteur — ADM1",
                    fontsize=12, pad=10)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANNEAU 2 — Alcalinité
    # ═══════════════════════════════════════════════════════════════════════════

    # Aire IC total (fond)
    ax_alk.fill_between(t, S_IC, alpha=0.12, color=C_IC)
    l_ic,   = ax_alk.plot(t, S_IC,    color=C_IC,   lw=1.5,
                          label="$S_{IC}$ carbone inorganique total  [kmol C/m³]")
    l_hco3, = ax_alk.plot(t, S_hco3,  color=C_HCO3, lw=2.0,
                          label="$S_{HCO_3^-}$ bicarbonate  [M]")
    l_co2,  = ax_alk.plot(t, S_co2_d, color=C_CO2,  lw=1.2, ls="--",
                          label="$S_{CO_2}$ CO₂ dissous  [M]")

    # Ratio HCO3/IC axe droit
    ax_alk_r.plot(t, ratio_hco3, color=C_RATIO_C, lw=1.2, ls="-.",
                  alpha=0.8, label="HCO₃⁻ / IC  [%]")
    ax_alk_r.axhline(80, color=C_RATIO_C, lw=0.7, ls=":", alpha=0.5)
    ax_alk_r.text(t_max * 0.01, 81,
                  "80 % (seuil tampon efficace)",
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
    _style_ax(ax_alk, ylabel="Concentration  [kmol/m³ ou M]", hide_x=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANNEAU 3 — Ratio VFA / alcalinité
    # ═══════════════════════════════════════════════════════════════════════════

    # Zones de stabilité
    ax_vfa_r.axhspan(0,    0.3,  color=Z_STABLE, alpha=0.55, zorder=0)
    ax_vfa_r.axhspan(0.3,  0.6,  color=Z_WARN,   alpha=0.55, zorder=0)
    ax_vfa_r.axhspan(0.6,  5.0,  color=Z_RISK,   alpha=0.40, zorder=0)

    # VFA total et alcalinité sur axe gauche
    ax_vfa.fill_between(t, VFA_total, alpha=0.15, color=C_VFA)
    l_vfa, = ax_vfa.plot(t, VFA_total, color=C_VFA, lw=1.8,
                         label="VFA totaux  (va+bu+pro+ac)  [M]")
    l_alk, = ax_vfa.plot(t, S_hco3,    color=C_ALK, lw=1.4, ls="--",
                         label="HCO₃⁻ (alcalinité)  [M]")

    # Ratio sur axe droit
    l_ratio, = ax_vfa_r.plot(t, ratio_vfa, color=C_RATIO_V, lw=2.0, zorder=5,
                              label="Ratio VFA/Alk  [—]")
    ax_vfa_r.axhline(0.3, color=C_RATIO_V, lw=0.8, ls=":", alpha=0.7)
    ax_vfa_r.axhline(0.6, color=C_RATIO_V, lw=0.8, ls=":", alpha=0.7)
    ax_vfa_r.text(t_max * 0.98, 0.31, "0.3  stable",
                  ha="right", fontsize=7.5, color=C_RATIO_V, alpha=0.8)
    ax_vfa_r.text(t_max * 0.98, 0.61, "0.6  risque",
                  ha="right", fontsize=7.5, color=C_RATIO_V, alpha=0.8)
    ax_vfa_r.set_ylabel("Ratio VFA / alcalinité  [—]", fontsize=9,
                         rotation=-90, labelpad=14, color=C_RATIO_V)
    ax_vfa_r.tick_params(axis="y", labelcolor=C_RATIO_V)
    ax_vfa_r.set_ylim(0, max(1.5, np.nanpercentile(ratio_vfa, 98) * 1.3))
    ax_vfa_r.spines[["top", "left"]].set_visible(False)

    handles_vfa = [l_vfa, l_alk,
                   plt.Line2D([0],[0], color=C_RATIO_V, lw=2,
                              label="Ratio VFA / Alk  [—]"),
                   mpatches.Patch(color=Z_STABLE, label="Stable  (< 0.3)"),
                   mpatches.Patch(color=Z_WARN,   label="Attention  (0.3–0.6)"),
                   mpatches.Patch(color=Z_RISK,   label="Risque  (> 0.6)")]
    ax_vfa.legend(handles=handles_vfa, fontsize=8.5, loc="upper right",
                  framealpha=0.85, ncol=2)
    _style_ax(ax_vfa, ylabel="Concentration  [M]", hide_x=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANNEAU 4 — Azote inorganique et NH3 libre
    # ═══════════════════════════════════════════════════════════════════════════

    # S_IN axe gauche (kmol/m³)
    ax_n.fill_between(t, S_IN, alpha=0.10, color=C_IN)
    l_in, = ax_n.plot(t, S_IN, color=C_IN, lw=1.4, ls="--",
                      label="$S_{IN}$ azote inorganique total  [kmol N/m³]")

    # NH3 et NH4 axe droit (M)
    l_nh4, = ax_n_r.plot(t, S_nh4, color=C_NH4, lw=1.2, ls="--", alpha=0.7,
                          label="$S_{NH_4^+}$ ammonium  [M]")
    l_nh3, = ax_n_r.plot(t, S_nh3, color=C_NH3, lw=2.2,
                          label="$S_{NH_3}$ ammoniac libre  [M]")

    # Seuil d'inhibition K_I_nh3
    ax_n_r.axhline(K_I_nh3, color="#c0392b", lw=1.0, ls="--", alpha=0.85)
    ax_n_r.text(t_max * 0.01, K_I_nh3 * 1.06,
                f"$K_{{I,NH_3}}$ = {K_I_nh3:.4f} M  (seuil inhibition)",
                fontsize=8, color="#c0392b", alpha=0.9)

    # Annotation : % du temps où NH3 dépasse le seuil
    over_threshold = np.sum(S_nh3 > K_I_nh3)
    pct_over = 100 * over_threshold / max(len(S_nh3), 1)
    if pct_over > 0:
        ax_n_r.text(0.98, 0.92,
                    f"NH₃ > seuil : {pct_over:.1f} % du temps",
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
                            label=f"Seuil $K_{{I,NH_3}}$ = {K_I_nh3:.4f} M")]
    ax_n.legend(handles=handles_n, fontsize=8.5, loc="upper right",
                framealpha=0.85)
    _style_ax(ax_n, ylabel="$S_{IN}$  [kmol N/m³]",
              xlabel="Temps  [j]", hide_x=False)

    # ── Finitions globales ────────────────────────────────────────────────────
    fig.align_ylabels([ax_ph, ax_alk, ax_vfa, ax_n])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graphique pH/alcalinité sauvegardé → {save_path}")

    if show:
        plt.show()

    return fig


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _style_ax(ax, ylabel="", xlabel="", ylim=None, hide_x=False):
    """Applique un style cohérent à un axe."""
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
"""
Autor :  Margaux Bonal 
Email : margaux.bonal@inrae.fr
Date : 04/2026

ADM1 — Graphique : Évolution des biomasses microbiennes
=========================================================
Trois panneaux sur une même figure :

  Panneau haut   : Aires empilées NORMALISÉES (% de la biomasse totale active)
                   → montre l'évolution des parts relatives de chaque population
                   → permet de voir les changements de dominance sans être
                     masqué par les variations d'amplitude absolue

  Panneau milieu : Aires empilées en valeurs ABSOLUES [kgCOD/m³]
                   → montre la biomasse totale et son évolution quantitative

  Panneau bas    : Taux de croissance net apparent de chaque population [d⁻¹]
                   calculé comme  (X(t) - X(t-1)) / X(t-1) / dt
                   → signal d'alarme précoce : une population en déclin rapide
                     annonce un problème avant que le biogaz ne chute

Les 7 populations microbiennes ADM1 :
  X_su  : dégradeurs de sucres          (acidogénèse)
  X_aa  : dégradeurs d'acides aminés    (acidogénèse)
  X_fa  : dégradeurs de LCFA            (acétogénèse)
  X_c4  : dégradeurs de valérate/butyrate (acétogénèse)
  X_pro : dégradeurs de propionate      (acétogénèse — souvent limitant)
  X_ac  : méthanogènes acétoclastes     (méthanogénèse — ~70% CH4)
  X_h2  : méthanogènes hydrogénophiles  (méthanogénèse — ~30% CH4)

Utilisation :
    from plots.plot_biomass import plot_biomass
    plot_biomass(df, save_path="results/figures/biomass.png", show=True)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# ── Palette des 7 populations ─────────────────────────────────────────────────
# Organisation par étape biochimique : chaud → froid au fil de la chaîne
POPULATIONS = {
    "X_su":  {"label": "$X_{su}$  sucres",          "color": "#e07b39", "step": "Acidogénèse"},
    "X_aa":  {"label": "$X_{aa}$  acides aminés",   "color": "#c0392b", "step": "Acidogénèse"},
    "X_fa":  {"label": "$X_{fa}$  LCFA",            "color": "#8e44ad", "step": "Acétogénèse"},
    "X_c4":  {"label": "$X_{c4}$  C4 (val/but)",   "color": "#6c5ce7", "step": "Acétogénèse"},
    "X_pro": {"label": "$X_{pro}$  propionate",     "color": "#2980b9", "step": "Acétogénèse"},
    "X_ac":  {"label": "$X_{ac}$  acétate (méth.)", "color": "#27ae60", "step": "Méthanogénèse"},
    "X_h2":  {"label": "$X_{h2}$  H₂ (méth.)",     "color": "#16a085", "step": "Méthanogénèse"},
}

# Couleurs de fond par étape pour les annotations
STEP_COLORS = {
    "Acidogénèse":   "#fff0e6",
    "Acétogénèse":   "#f0eaff",
    "Méthanogénèse": "#e6f7f2",
}


def _growth_rate(x_arr, t_arr):
    """
    Taux de croissance net apparent  Δ(ln X) / Δt  [d⁻¹].
    Retourne un tableau de même longueur (NaN au premier point).
    """
    dt   = np.diff(t_arr)
    dx   = np.diff(np.log(np.maximum(x_arr, 1e-12)))
    rate = np.where(dt > 0, dx / dt, np.nan)
    return np.concatenate([[np.nan], rate])


def plot_biomass(df, save_path: str = None, show: bool = True):
    """
    Paramètres
    ----------
    df         : DataFrame pandas avec colonnes ADM1 + colonne 'time'
    save_path  : chemin de sauvegarde (PNG/PDF) — None = pas de sauvegarde
    show       : afficher la fenêtre matplotlib
    """
    t    = df["time"].values
    keys = list(POPULATIONS.keys())

    # ── Extraction des données ────────────────────────────────────────────────
    data = {k: df[k].values for k in keys}

    # Biomasse totale active
    X_tot = np.sum([data[k] for k in keys], axis=0)
    X_tot_safe = np.where(X_tot > 0, X_tot, np.nan)

    # Parts relatives (%)
    pct = {k: 100 * data[k] / X_tot_safe for k in keys}

    # Taux de croissance net
    rates = {k: _growth_rate(data[k], t) for k in keys}

    # Valeurs empilées pour les aires absolues
    stack_abs = np.array([data[k] for k in keys])
    stack_pct = np.array([pct[k]  for k in keys])

    colors = [POPULATIONS[k]["color"] for k in keys]
    labels = [POPULATIONS[k]["label"] for k in keys]

    # ── Mise en page ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor("white")

    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[3, 2.5, 2],
                  hspace=0.10)

    ax_pct   = fig.add_subplot(gs[0])
    ax_abs   = fig.add_subplot(gs[1], sharex=ax_pct)
    ax_rate  = fig.add_subplot(gs[2], sharex=ax_pct)

    # ── Panneau 1 : aires normalisées ─────────────────────────────────────────
    ax_pct.stackplot(t, stack_pct, labels=labels, colors=colors, alpha=0.82)

    # Ligne de référence : part des méthanogènes (X_ac + X_h2)
    pct_meth = pct["X_ac"] + pct["X_h2"]
    ax_pct.plot(t, pct_meth, color="white", lw=1.2, ls="--", alpha=0.7,
                zorder=5)

    # Annotation bandes par étape (en fond, premier panneau)
    _add_step_bands(ax_pct, stack_pct, keys)

    ax_pct.set_ylim(0, 100)
    ax_pct.set_ylabel("Part relative  [%]", fontsize=10)
    ax_pct.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax_pct.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
    ax_pct.set_title(
        "Évolution des populations microbiennes — ADM1", fontsize=12, pad=10
    )

    # Légende compacte à droite
    _add_legend(ax_pct, keys, colors, labels)

    # Annotation ligne pointillée méthanogènes
    ax_pct.text(
        t[len(t) // 10], float(np.nanmean(pct_meth)) + 2.5,
        "── méthanogènes totaux",
        color="white", fontsize=8, alpha=0.85,
    )

    plt.setp(ax_pct.get_xticklabels(), visible=False)

    # ── Panneau 2 : aires absolues ────────────────────────────────────────────
    ax_abs.stackplot(t, stack_abs, colors=colors, alpha=0.78)

    # Courbe biomasse totale (axe droit)
    ax_abs_r = ax_abs.twinx()
    ax_abs_r.plot(t, X_tot, color="#2c3e50", lw=1.6, ls="-",
                  label="Biomasse totale")
    ax_abs_r.set_ylabel("Total  [kgCOD/m³]", fontsize=9,
                         rotation=-90, labelpad=14, color="#2c3e50")
    ax_abs_r.tick_params(axis="y", labelcolor="#2c3e50")
    ax_abs_r.spines[["top", "left"]].set_visible(False)

    ax_abs.set_ylabel("Biomasse  [kgCOD/m³]", fontsize=10)
    ax_abs.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
    ax_abs.spines[["top", "right"]].set_visible(False)
    plt.setp(ax_abs.get_xticklabels(), visible=False)

    # ── Panneau 3 : taux de croissance net ────────────────────────────────────
    ax_rate.axhline(0, color="#555", lw=0.8, ls="-", alpha=0.4)

    # Zone de déclin : rouge pâle en dessous de 0
    ax_rate.axhspan(-999, 0, color="#fdecea", alpha=0.35, zorder=0)
    # Zone de croissance : verte pâle au-dessus de 0
    ax_rate.axhspan(0, 999, color="#e8f5ee", alpha=0.35, zorder=0)

    for k in keys:
        r = rates[k]
        # Lissage léger (moyenne mobile 7 jours) pour lisibilité
        r_smooth = _moving_avg(r, window=7)
        ax_rate.plot(t, r_smooth,
                     color=POPULATIONS[k]["color"],
                     lw=1.2, alpha=0.85,
                     label=POPULATIONS[k]["label"])

    # Limiter l'axe Y pour éviter les artefacts du démarrage
    valid = []
    for k in keys:
        v = rates[k][~np.isnan(rates[k])]
        if len(v):
            valid.extend(v.tolist())
    if valid:
        p5, p95 = np.percentile(valid, 5), np.percentile(valid, 95)
        margin = max(abs(p5), abs(p95)) * 1.5
        ax_rate.set_ylim(-margin, margin)

    ax_rate.set_ylabel("Taux croissance net  [d⁻¹]", fontsize=10)
    ax_rate.set_xlabel("Temps  [j]", fontsize=10)
    ax_rate.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
    ax_rate.spines[["top", "right"]].set_visible(False)

    # Annotation zones
    ax_rate.text(t[3], ax_rate.get_ylim()[1] * 0.85,
                 "croissance", color="#27ae60", fontsize=8, alpha=0.8)
    ax_rate.text(t[3], ax_rate.get_ylim()[0] * 0.85,
                 "déclin", color="#c0392b", fontsize=8, alpha=0.8)

    # ── Alignement labels Y ───────────────────────────────────────────────────
    fig.align_ylabels([ax_pct, ax_abs, ax_rate])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graphique biomasses sauvegardé → {save_path}")

    if show:
        plt.show()

    return fig


# ── Fonctions utilitaires ──────────────────────────────────────────────────────

def _add_step_bands(ax, stack_pct, keys):
    """
    Ajoute des étiquettes d'étape sur le bord droit du panneau normalisé,
    positionnées au centre de la bande de couleur correspondante.
    Acidogénèse = X_su + X_aa  |  Acétogénèse = X_fa + X_c4 + X_pro
    Méthanogénèse = X_ac + X_h2
    """
    step_groups = {
        "Acidogénèse":   ["X_su", "X_aa"],
        "Acétogénèse":   ["X_fa", "X_c4", "X_pro"],
        "Méthanogénèse": ["X_ac", "X_h2"],
    }
    step_colors_text = {
        "Acidogénèse":   "#7f3b08",
        "Acétogénèse":   "#4a235a",
        "Méthanogénèse": "#0b5345",
    }

    # Calcul des cumulatifs pour positionner les étiquettes à t_mid
    t_mid_idx = len(stack_pct[0]) // 2
    cumul = 0.0
    for step_name, step_keys in step_groups.items():
        idx_list = [list(POPULATIONS.keys()).index(k) for k in step_keys]
        band_height = sum(stack_pct[i][t_mid_idx] for i in idx_list)
        y_center = cumul + band_height / 2
        cumul += band_height

        if band_height > 3:   # n'affiche l'étiquette que si la bande est visible
            ax.text(
                ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 1,
                y_center,
                step_name,
                ha="right", va="center",
                fontsize=8, color=step_colors_text[step_name],
                fontweight="bold", alpha=0.85,
            )


def _add_legend(ax, keys, colors, labels):
    """Légende compacte positionnée hors de la zone de tracé."""
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
    """Moyenne mobile simple, gère les NaN et les bords."""
    result = np.full_like(arr, np.nan, dtype=float)
    valid  = ~np.isnan(arr)
    if not valid.any():
        return result
    # Remplace NaN par 0 le temps du calcul
    filled = np.where(valid, arr, 0.0)
    kernel = np.ones(window) / window
    smooth = np.convolve(filled, kernel, mode="same")
    # Correction des bords (moins de points disponibles)
    count  = np.convolve(valid.astype(float), kernel * window, mode="same")
    count  = np.where(count > 0, count, np.nan)
    smooth = smooth / (count / window)
    result[valid] = smooth[valid]
    return result
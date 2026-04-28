"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Main entry point — simulation runner
 
Authors:
    - Margaux Bonal           <margaux.bonal@inrae.fr>
    - David Camilo Corrales   <David-Camilo.Corrales-Munoz@inrae.fr>
Date: 04/2026
 
Configuration files expected at:
    configs/
    ├── adm1_parameters.yaml     ← intrinsic ADM1 kinetic/stoichiometric parameters
    ├── Initial_states.yaml      ← initial state vectors (one per named scenario)
    ├── Influent.yaml            ← influent definition (dynamic CSV or constant values)
    ├── Scenario.yaml            ← active scenario selector + parameter overrides
    └── Simulation.yaml          ← ODE solver settings, time horizon, output options
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.reactor import ADM1Reactor, FULL_STATE_NAMES
from src.parameters import ADM1Parameters
from initial_states import InitialState
from src.influent import Influent

from src.acid_base import compute_acid_base_equilibrium, compute_total_cod

from plots.plot_biogas import plot_biogas
from plots.plot_biomass import plot_biomass
from plots.plot_pH_alkalinity import plot_pH_alkalinity

def main():
    # ============================================================
    # 0. Chemins de configuration
    # ============================================================
    PARAMS_FILE = "configs/adm1_parameters.yaml"
    STATES_FILE = "configs/Initial_states.yaml"
    INFLUENT_FILE = "configs/Influent.yaml"
    SCENARIO_FILE = "configs/Scenario.yaml"
    SIMULATION_FILE = "configs/Simulation.yaml"


    # ============================================================
    # 1. Outils utilitaires
    # ============================================================
    def load_yaml_file(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Le fichier YAML {path} doit contenir un dictionnaire.")
        return data


    def safe_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)


    def build_time_vector(t_start: float, t_end: float, dt_out: float) -> np.ndarray:
        """
        Build the array of time points at which ODE results are recorded.
 
        The vector starts at t_start, increments by dt_out, and always
        includes t_end as the last point (even if it falls between two steps).
 
        Parameters
        ----------
        t_start : float  Simulation start time (d).
        t_end   : float  Simulation end time (d).
        dt_out  : float  Output time step (d). Must be > 0.
 
        Returns
        -------
        np.ndarray : 1-D array of time evaluation points (d).
        """
        if dt_out <= 0:
            raise ValueError("dt_out doit être strictement positif.")
        if t_end <= t_start:
            raise ValueError("t_end doit être strictement supérieur à t_start.")

        # inclut explicitement le point final si possible
        n_steps = int(np.floor((t_end - t_start) / dt_out)) + 1
        t_eval = t_start + np.arange(n_steps) * dt_out

        # Append t_end explicitly so the final state is always captured
        if t_eval[-1] < t_end:
            t_eval = np.append(t_eval, t_end)

        return t_eval


    def maybe_print(msg: str, enabled: bool = True):
        if enabled:
            print(msg)


    # ============================================================
    # 2. Chargement de la configuration
    # ============================================================
    sim_cfg = load_yaml_file(SIMULATION_FILE)

    solver_cfg = sim_cfg.get("solver", {})
    time_cfg = sim_cfg.get("time", {})
    output_cfg = sim_cfg.get("output", {})
    verbose_cfg = sim_cfg.get("verbose", {})

    verbose_enabled = bool(verbose_cfg.get("enabled", verbose_cfg.get("print_time", True)))
    print_banner = bool(verbose_cfg.get("print_startup_banner", True))
    print_interval_days = int(verbose_cfg.get("print_interval_days", verbose_cfg.get("print_interval", 10)))


    # ============================================================
    # 3. Chargement paramètres / états initiaux / influent
    # ============================================================
    # Load intrinsic ADM1 parameters (kinetic, stoichiometric, physico-chemical)
    param = ADM1Parameters(
        params_file=PARAMS_FILE,
        scenarios_file=SCENARIO_FILE,
    )

    # Load initial state vector y0 (38 ADM1 state variables)
    init_state = InitialState(
        states_file=STATES_FILE,
        scenarios_file=SCENARIO_FILE,
    )
    y0_full = np.asarray(init_state.get_vector(), dtype=float)

    # Load influent data (either time-series from CSV or constant values from YAML)
    influent = Influent(
        influent_file=INFLUENT_FILE,
        scenarios_file=SCENARIO_FILE,
    )

    # Instantiate the ADM1 reactor model (holds all ODE equations and process rates)
    reactor = ADM1Reactor(param, constants=None)

    # Reprojette l'état initial sur l'équilibre acido-basique pour démarrer
    # avec pH / HCO3- / CO2 / NH3 / NH4+ cohérents.
    initial_state_dict = reactor.unpack_state(y0_full)
    initial_acid_base = compute_acid_base_equilibrium(initial_state_dict, param)
    y0_full = reactor._apply_acid_base_to_state_vector(y0_full, initial_acid_base)
    y0 = reactor.reduce_to_dynamic_state(y0_full)


    # ============================================================
    # 4. Construction du temps de simulation
    # ============================================================
    time_data = np.asarray(influent.get_time(), dtype=float) # time axis of the influent data (d)

    if len(time_data) == 0:
        raise ValueError("Influent vide : aucun point de temps trouvé.")

    t_start = safe_float(time_cfg.get("t_start", 0), 0)

    t_end_cfg = time_cfg.get("t_end", None)

    if t_end_cfg is None:
        # si influent constant ou temps court, on impose une durée par défaut raisonnable
        t_end = safe_float(t_end_cfg, 365.0)
    
    elif getattr(influent, "mode", "constant") == "dynamic":
        # Dynamic influent: simulation duration = length of the influent time series
        # This ensures we don't simulate beyond available input data
        t_end = float(time_data[-1])
 
    else:
        # Constant influent: no natural end time from data.
        # Use t_end_default_constant from config, or fall back to 365 d.
        # Rule of thumb: simulate at least 3-5× the hydraulic retention time (HRT)
        # to ensure the reactor reaches steady state.
        t_end_default = safe_float(time_cfg.get("t_end_default_constant", 365.0), 365.0)
        t_end = t_end_default


    dt_out = safe_float(time_cfg.get("dt_out", 1.0), 1.0) # output time step (d), default = 1 day
    t_eval = build_time_vector(t_start, t_end, dt_out)


    # ============================================================
    # 5. Préchargement de l'influent pour éviter les appels répétés
    # ============================================================
    # The ODE solver calls ADM1_wrapper thousands of times per simulation day.
    # Reading the influent object once per day step and caching results in a list
    # avoids repeated dictionary lookups / interpolations inside the hot loop.
    influent_cache = [influent.get(i) for i in range(len(time_data))]


    def get_influent_for_time(t: float) -> dict:
        """
        Return the influent state dict corresponding to simulation time t.

        Uses np.searchsorted against the actual influent time axis to find the
        active step (zero-order hold). This is robust to time axes that don't
        start at 0 or aren't spaced exactly one day apart.

        Parameters
        ----------
        t : float  Current simulation time (d).

        Returns
        -------
        dict : Influent concentrations keyed as 'S_su_in', 'X_xc_in', etc. (kgCOD/m³)
        """
        step = int(np.searchsorted(time_data, t, side="right") - 1)
        step = max(0, min(step, len(influent_cache) - 1))
        return influent_cache[step]


    # ============================================================
    # 6. Bannière de démarrage
    # ============================================================
    def print_startup_banner():
        """
        Print a human-readable summary of the simulation setup to the console.
        Helps quickly verify that the correct scenario, reactor volume, and
        solver settings are being used before committing to a long run.
        """
        scenario_cfg = load_yaml_file(SCENARIO_FILE)
        active_key = scenario_cfg.get("active_scenario", "?")
        scenario = scenario_cfg.get("scenarios", {}).get(active_key, {})
        sc_desc = str(scenario.get("description", "")).replace("\n", " ").strip()

        # Compute hydraulic retention time HRT = V_liq / q_ad (d)
        try:
            trh = param.V_liq / param.q_ad
        except Exception:
            trh = float("nan")

        # Classify operating temperature regime
        T_op_C = param.T_op - 273.15    # convert from Kelvin to Celsius
        if T_op_C < 20:
            regime = "psychrophile"
        elif T_op_C < 42:
            regime = "mésophile"
        elif T_op_C < 55:
            regime = "intermédiaire"
        else:
            regime = "thermophile"

        inf0 = get_influent_for_time(0.0)

        print("\n" + "━" * 66)
        print("  ADM1 — Configuration de la simulation")
        print("━" * 66)

        print("\n  ── Scénario ────────────────────────────────────────────────────")
        print(f"  Nom actif   : {active_key}")
        if sc_desc:
            print(f"  Description : {sc_desc}")

        print("\n  ── Réacteur ────────────────────────────────────────────────────")
        print(f"  Volume liquide (V_liq) : {param.V_liq:.1f} m³")
        print(f"  Volume gazeux  (V_gas) : {param.V_gas:.1f} m³")
        print(f"  Débit influent (q_ad)  : {param.q_ad:.1f} m³/j")
        print(f"  TRH                   : {trh:.1f} j")
        print(f"  Température           : {T_op_C:.1f} °C ({regime})")

        print("\n  ── Influent ────────────────────────────────────────────────────")
        print(f"  Mode : {getattr(influent, 'mode', 'inconnu')}")
        print(f"  S_su_in : {inf0.get('S_su_in', float('nan')):.4f} kgCOD/m³")
        print(f"  S_aa_in : {inf0.get('S_aa_in', float('nan')):.4f} kgCOD/m³")
        print(f"  X_xc_in : {inf0.get('X_xc_in', float('nan')):.4f} kgCOD/m³")
        print(f"  S_IC_in : {inf0.get('S_IC_in', float('nan')):.4f} kmolC/m³")
        print(f"  S_IN_in : {inf0.get('S_IN_in', float('nan')):.4f} kmolN/m³")

        print("\n  ── Solveur numérique ───────────────────────────────────────────")
        print(f"  Méthode      : {solver_cfg.get('method', 'BDF')}")
        print(f"  rtol / atol  : {solver_cfg.get('rtol', 1e-5):.0e} / {solver_cfg.get('atol', 1e-7):.0e}")
        print(f"  Durée simulée: {t_start:.1f} → {t_end:.1f} j")
        print(f"  Pas de sortie: {dt_out:.2f} j/point")
        print("━" * 66 + "\n")


    if print_banner:
        print_startup_banner()


    # ============================================================
    # 7. Résumé journalier
    # ============================================================
    _last_printed_bucket = [-1]  # mutable container so the nested function can update it
    _wall_start = [0.0]          # wall-clock time at simulation start, for elapsed time display
    _max_t_seen = [float("-inf")]
    _last_progress_wall = [0.0]


    def print_day_summary(t_day: float, summary: dict, elapsed_wall: float):
        def flag_inhib(val, seuil_warn=0.5, seuil_crit=0.2):
            if val < seuil_crit:
                return "⚠ fort"
            if val < seuil_warn:
                return "~ modéré"
            return "✓ ok"

        print("\n" + "━" * 62)
        print(f"  Jour {t_day:>6.1f} d   |   wall-clock : {elapsed_wall:>6.1f} s")
        print("─" * 62)
        print(f"  pH             : {summary.get('pH', float('nan')):8.3f}")
        print(f"  Débit biogaz   : {summary.get('q_gas', float('nan')):8.2f} m³/d")
        print(f"  P CH₄ (gaz)    : {summary.get('p_gas_ch4_bar', float('nan')):8.4f} bar")
        print("  Inhibitions :")
        print(f"    pH acidogènes  {summary.get('I_pH_aa', float('nan')):6.3f}  {flag_inhib(summary.get('I_pH_aa', 1.0))}")
        print(f"    NH₃ acétogènes {summary.get('I_nh3', float('nan')):6.3f}  {flag_inhib(summary.get('I_nh3', 1.0))}")
        print(f"    H₂ LCFA        {summary.get('I_h2_fa', float('nan')):6.3f}  {flag_inhib(summary.get('I_h2_fa', 1.0))}")
        print("  DCO :")
        print(f"  DCO liquide     : {summary.get('COD_liquid_total', float('nan')):8.3f} kgCOD/m³")
        print(f"  Stock total     : {summary.get('COD_total_inventory', float('nan')):8.1f} kgCOD")

        print("━" * 62)


    # ============================================================
    # 8. Wrapper solveur
    # ============================================================
    def ADM1_wrapper(t, y):
        # Inject the current influent into the reactor (updates feed concentrations)
        reactor.influent_state = get_influent_for_time(t)
        # Compute all 38 differential equations of the ADM1 model
        dydt = reactor.ADM1_ODE(t, y)

        if t > _max_t_seen[0]:
            _max_t_seen[0] = float(t)

        # Periodic console logging — avoids flooding the terminal on every solver step
        if verbose_enabled and hasattr(reactor, "get_process_summary"):
            progress_t = _max_t_seen[0]
            bucket = int(np.floor(progress_t / max(print_interval_days, 1)))
            if bucket > _last_printed_bucket[0]:
                _last_printed_bucket[0] = bucket
                try:
                    summary = reactor.get_process_summary()
                    elapsed_wall = time.time() - _wall_start[0]
                    _last_progress_wall[0] = elapsed_wall
                    print_day_summary(progress_t, summary, elapsed_wall)
                except Exception:
                    pass
            elif _wall_start[0] > 0.0:
                elapsed_wall = time.time() - _wall_start[0]
                if elapsed_wall - _last_progress_wall[0] >= 30.0:
                    _last_progress_wall[0] = elapsed_wall
                    maybe_print(
                        f"  Progression solveur : t_max atteint = {_max_t_seen[0]:.3f} j | wall-clock : {elapsed_wall:>6.1f} s",
                        verbose_enabled,
                    )

        return dydt


    # ============================================================
    # 9. Résolution ODE
    # ============================================================
    maybe_print("━" * 62, verbose_enabled)
    maybe_print(f"  Simulation ADM1 — méthode {solver_cfg.get('method', 'BDF')}", verbose_enabled)
    maybe_print(f"  t_start={t_start:.1f} d  →  t_end={t_end:.1f} d  (dt_out={dt_out} d)", verbose_enabled)
    maybe_print("━" * 62 + "\n", verbose_enabled)

    t0 = time.time()
    _wall_start[0] = t0

    sol = solve_ivp(
        fun=ADM1_wrapper,
        t_span=(t_start, t_end),
        y0=y0,
        t_eval=t_eval,
        method=solver_cfg.get("method", "BDF"),
        rtol=solver_cfg.get("rtol", 1e-5),
        atol=solver_cfg.get("atol", 1e-7),
        max_step=solver_cfg.get("max_step", 0.5),
        dense_output=solver_cfg.get("dense_output", False),
    )

    elapsed = time.time() - t0
    print(f"\nTemps de simulation : {int(elapsed // 60)} min {int(elapsed % 60)} s")
    print(f"Succès : {sol.success} | Message : {sol.message}")

    if not sol.success:
        raise RuntimeError(f"Échec solveur : {sol.message}")


    # ============================================================
    # 10. Sauvegarde des résultats
    # ============================================================
    reconstructed_rows = []
    for dynamic_row in sol.y.T:
        full_state_vector = reactor.expand_dynamic_state(dynamic_row)
        state = reactor.unpack_state(full_state_vector)
        acid_base = compute_acid_base_equilibrium(state, param)
        reconstructed_rows.append(reactor._apply_acid_base_to_state_vector(full_state_vector, acid_base))

    df = pd.DataFrame(reconstructed_rows, columns=FULL_STATE_NAMES)
    df.insert(0, "time", sol.t)

    acid_base_rows = []
    cod_rows = []

    for _, row in df.iterrows():
        state = {name: float(row[name]) for name in FULL_STATE_NAMES}
        acid_base_rows.append(compute_acid_base_equilibrium(state, param))
        cod_rows.append(compute_total_cod(state, param))

    acid_base_df = pd.DataFrame(acid_base_rows)
    cod_df = pd.DataFrame(cod_rows)

    for col in acid_base_df.columns:
        df[col] = acid_base_df[col]

    for col in cod_df.columns:
        df[col] = cod_df[col]

    output_dir = output_cfg.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    if output_cfg.get("save_results", True):
        out_path = os.path.join(output_dir, output_cfg.get("filename", "dynamic_out.csv"))
        df.to_csv(out_path, index=False)
        print(f"\nRésultats sauvegardés → {out_path} ✅")


    # ============================================================
    # 11. Graphiques robustes
    # ============================================================
    save_figures = bool(output_cfg.get("save_figures", True))
    show_figures = bool(output_cfg.get("show_figures", True))
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\nGénération des graphiques...")

    n_points = len(df)

    if n_points < 2:
        print("Trop peu de points pour générer des graphiques fiables. Graphes ignorés.")
    else:
        try:
            plot_biogas(
                df=df,
                param=param,
                save_path=os.path.join(fig_dir, "biogas.png") if save_figures else None,
                show=show_figures,
            )
            print("Graphique biogaz généré ✅")
        except Exception as e:
            print(f"Graphique biogaz ignoré : {e}")

        try:
            plot_biomass(
                df=df,
                save_path=os.path.join(fig_dir, "biomass.png") if save_figures else None,
                show=show_figures,
            )
            print("Graphique biomasse généré ✅")
        except Exception as e:
            print(f"Graphique biomasse ignoré : {e}")

        try:
            plot_pH_alkalinity(
                df=df,
                param=param,
                save_path=os.path.join(fig_dir, "pH_alkalinity.png") if save_figures else None,
                show=show_figures,
            )
            print("Graphique pH/alcalinité généré ✅")
        except Exception as e:
            print(f"Graphique pH/alcalinité ignoré : {e}")

        # Visualisation rapide seulement si assez de points
        try:
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

            axes[0].plot(df["time"], df["S_ac"], label="Acétate (S_ac)")
            axes[0].set_ylabel("kgCOD.m⁻³")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(df["time"], df["S_ch4"], label="Méthane (S_ch4)")
            axes[1].set_ylabel("kgCOD.m⁻³")
            axes[1].legend()
            axes[1].grid(True)

            axes[2].plot(df["time"], df["S_h2"], label="Hydrogène (S_h2)")
            axes[2].set_ylabel("kgCOD.m⁻³")
            axes[2].set_xlabel(f"Temps [{time_cfg.get('units', 'd')}]")
            axes[2].legend()
            axes[2].grid(True)

            fig.suptitle("Simulation ADM1", fontsize=13)
            plt.tight_layout()
            if show_figures:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            print(f"Visualisation rapide ignorée : {e}")

if __name__ == "__main__":
    main()

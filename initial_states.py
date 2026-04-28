"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: InitialState

Author: Margaux Bonal
Email: margaux.bonal@inrae.fr
Date: 04/2026

Charge le vecteur d'état initial depuis :
  configs/Initial_states.yaml  (jeu d'états nommés)
  configs/Scenario.yaml        (sélection du jeu actif)

Le vecteur y0 est retourné dans l'ordre canonique des 38 variables ADM1.
"""

import numpy as np
import yaml
from src.acid_base import compute_required_strong_ion_for_pH
from src.parameters import ADM1Parameters


# Ordre canonique des 38 variables d'état ADM1
STATE_VARIABLES = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac",
    "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
    "X_xc", "X_ch", "X_pr", "X_li",
    "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
    "S_cation", "S_anion",
    "S_H_ion", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion",
    "S_hco3_ion", "S_co2", "S_nh3", "S_nh4_ion",
    "S_gas_h2", "S_gas_ch4", "S_gas_co2",
]


class InitialState:
    """
    Charge et expose le vecteur d'état initial y0 pour le solveur ODE.

    Usage :
        init = InitialState()
        y0 = init.get_vector()
    """

    def __init__(
        self,
        states_file: str = "configs/Initial_states.yaml",
        scenarios_file: str = "configs/Scenario.yaml",
    ):
        self.states_file = states_file
        self.scenarios_file = scenarios_file
        self.state_dict: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def _load(self):
        """Détermine le jeu d'états actif et le charge."""
        active_key = self._get_active_key()
        with open(self.states_file, "r") as f:
            raw = yaml.safe_load(f)

        if active_key not in raw:
            available = list(raw.keys())
            raise KeyError(
                f"Jeu d'états initiaux '{active_key}' introuvable dans "
                f"{self.states_file}. Disponibles : {available}"
            )

        state_set = raw[active_key]
        self.state_dict = {
            k: float(v["value"])
            for k, v in state_set.items()
            if isinstance(v, dict) and "value" in v
        }

        target_pH = None
        if isinstance(state_set.get("pH"), dict) and "value" in state_set["pH"]:
            target_pH = float(state_set["pH"]["value"])

        if target_pH is not None:
            self.state_dict["S_H_ion"] = 10.0 ** (-target_pH)
            param = ADM1Parameters(scenarios_file=self.scenarios_file)
            self.state_dict["S_anion"] = compute_required_strong_ion_for_pH(
                self.state_dict,
                param,
                target_pH=target_pH,
                solve_for="S_anion",
            )

    def _get_active_key(self) -> str:
        """Lit le scénario actif pour en extraire la clé d'états initiaux."""
        try:
            with open(self.scenarios_file, "r") as f:
                raw = yaml.safe_load(f)
            active_scenario = raw.get("active_scenario", "BSM2")
            key = (
                raw.get("scenarios", {})
                .get(active_scenario, {})
                .get("initial_states", "BSM2")
            )
            return key
        except FileNotFoundError:
            return "BSM2"

    # ------------------------------------------------------------------
    # Accès
    # ------------------------------------------------------------------

    def get_vector(self) -> np.ndarray:
        """
        Retourne le vecteur y0 dans l'ordre canonique des 38 variables ADM1.
        Lève une erreur si une variable est manquante dans le fichier YAML.
        """
        missing = [v for v in STATE_VARIABLES if v not in self.state_dict]
        if missing:
            raise KeyError(
                f"Variables manquantes dans l'état initial : {missing}"
            )
        return np.array([self.state_dict[v] for v in STATE_VARIABLES])

    def get_dict(self) -> dict:
        """Retourne le dictionnaire complet des états initiaux."""
        return dict(self.state_dict)

    def __repr__(self) -> str:
        return (
            f"<InitialState: {len(self.state_dict)} variables, "
            f"{len(STATE_VARIABLES)} dans le vecteur ODE>"
        )

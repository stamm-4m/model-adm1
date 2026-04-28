"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: Influent

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026 


—  Réorganisé par Margaux Bonal 
Email : margaux.bonal@inrae.fr
Date : 04/2026


Charge la configuration de l'influent depuis :
  configs/influent/influent.yaml      (définition des modes)
  configs/scenarios/scenarios.yaml    (sélection du mode actif)

Deux modes :
  - dynamic  : série temporelle issue d'un fichier CSV
  - constant : valeurs constantes définies dans influent.yaml
"""

import numpy as np
import pandas as pd
import yaml
from src.acid_base import compute_required_strong_ion_for_pH
from src.parameters import ADM1Parameters


# Variables d'entrée ADM1 (ordre de lecture du CSV pour le mode dynamique)
INFLUENT_VARS = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac",
    "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
    "X_xc", "X_ch", "X_pr", "X_li",
    "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
    "S_cation", "S_anion",
]


class Influent:
    """
    Gère les données d'entrée du digesteur.

    Usage :
        inf = Influent()
        inf.get(step)       -> dict avec clés "S_su_in", "S_aa_in", ...
        inf.get_time()      -> np.ndarray des instants (mode dynamique uniquement)
    """

    def __init__(
        self,
        influent_file: str = "configs/Influent.yaml",
        scenarios_file: str = "configs/Scenario.yaml",
    ):
        self.influent_file = influent_file
        self.scenarios_file = scenarios_file
        self.mode: str = ""
        self._time: np.ndarray | None = None
        self._data: pd.DataFrame | None = None          # mode dynamic
        self._constant_values: dict = {}                # mode constant
        self._load()

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def _load(self):
        """Lit la configuration et charge les données selon le mode actif."""
        mode_key = self._get_active_mode()

        with open(self.influent_file, "r") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(
                f"Le fichier {self.influent_file} est vide ou malformé "
                f"(yaml.safe_load a retourné : {type(raw).__name__})."
            )

        if mode_key is None or mode_key not in raw:
            available = list(raw.keys())
            raise KeyError(
                f"Mode d'influent '{mode_key}' introuvable dans "
                f"{self.influent_file}. Disponibles : {available}"
            )

        config = raw[mode_key]
        self.mode = config["type"]

        if self.mode == "dynamic":
            csv_path = config["file_path"]
            time_col = config.get("time_column", "time")
            self._data = pd.read_csv(csv_path)
            self._time = self._data[time_col].values

        elif self.mode == "constant":
            self._constant_values = {
                k: float(v["value"])
                for k, v in config.items()
                if isinstance(v, dict) and "value" in v
            }

            if isinstance(config.get("pH"), dict) and "value" in config["pH"]:
                target_pH = float(config["pH"]["value"])
                self._constant_values["S_H_ion"] = 10.0 ** (-target_pH)
                param = ADM1Parameters(scenarios_file=self.scenarios_file)
                self._constant_values["S_anion"] = compute_required_strong_ion_for_pH(
                    self._constant_values,
                    param,
                    target_pH=target_pH,
                    solve_for="S_anion",
                )

            # Temps fictif : [0, 1] pour permettre solve_ivp de démarrer
            self._time = np.array([0.0, 1.0])

        else:
            raise ValueError(f"Mode d'influent inconnu : '{self.mode}'")

    def _get_active_mode(self) -> str:
        """Lit le scénario actif et retourne la clé du mode d'influent."""
        try:
            with open(self.scenarios_file, "r") as f:
                raw = yaml.safe_load(f)

            if not isinstance(raw, dict):
                return "dynamic"

            active_scenario = raw.get("active_scenario") or "BSM2_dynamic"
            scenarios = raw.get("scenarios") or {}
            scenario = scenarios.get(active_scenario) or {}
            mode = scenario.get("influent_mode") or "dynamic"
            return mode

        except FileNotFoundError:
            return "dynamic"

    # ------------------------------------------------------------------
    # Accès
    # ------------------------------------------------------------------

    def get(self, step: int) -> dict:
        """
        Retourne un dictionnaire des variables d'influent au pas `step`.
        Les clés sont suffixées par '_in' (ex. "S_su_in").
        """
        if self.mode == "dynamic":
            step = min(step, len(self._time) - 1)
            return {var + "_in": self._data[var][step] for var in INFLUENT_VARS}
        else:
            return {var + "_in": self._constant_values.get(var, 0.0) for var in INFLUENT_VARS}

    def get_time(self) -> np.ndarray:
        """Retourne le vecteur de temps de la simulation."""
        return self._time

    def __repr__(self) -> str:
        n = len(self._time) if self._time is not None else 0
        return f"<Influent mode='{self.mode}' n_steps={n}>"

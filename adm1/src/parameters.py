"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: ADM1Parameters

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026 


—  Réorganisé par Margaux Bonal 
Email : margaux.bonal@inrae.fr
Date : 04/2026


Charge les paramètres du modèle ADM1 depuis la nouvelle architecture modulaire :
  - configs/adm1_parameters.yaml       : paramètres intrinsèques du modèle
  - configs/scenarios/scenarios.yaml   : scénario actif et surcharges
Les paramètres spécifiques à la simulation (T_op, q_ad, V_liq, etc.) peuvent être
surchargés par le scénario actif.
"""

import yaml


class ADM1Parameters:
    """
    Charge et expose les paramètres du modèle ADM1.

    Hiérarchie de résolution (priorité décroissante) :
        1. parameter_overrides du scénario actif
        2. Valeur T_op du scénario actif
        3. adm1_parameters.yaml (référentiel central)
    """

    def __init__(
        self,
        params_file: str = "configs/adm1_parameters_2.yaml",
        scenarios_file: str = "configs/Scenarios.yaml",
    ):
        self.params_file = params_file
        self.scenarios_file = scenarios_file
        self.params: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def _load(self):
        """Charge les paramètres de base puis applique les surcharges du scénario."""
        base = self._load_base_params()
        overrides = self._load_scenario_overrides()
        self.params = {**base, **overrides}

    def _load_base_params(self) -> dict:
        """Charge et aplatit adm1_parameters.yaml."""
        with open(self.params_file, "r") as f:
            raw = yaml.safe_load(f)
        return self._flatten(raw)

    def _load_scenario_overrides(self) -> dict:
        """
        Lit le scénario actif dans scenarios.yaml et retourne :
          - T_op du scénario (si présent)
          - tous les parameter_overrides du scénario
        """
        overrides = {}
        try:
            with open(self.scenarios_file, "r") as f:
                raw = yaml.safe_load(f)
        except FileNotFoundError:
            return overrides

        active_key = raw.get("active_scenario")
        if not active_key:
            return overrides

        scenario = raw.get("scenarios", {}).get(active_key, {})

        # Température opératoire spécifique au scénario
        t_op = scenario.get("T_op")
        if t_op:
            overrides["T_op"] = float(t_op["value"])
            # T_ad est un alias utilisé dans reactor.py
            overrides["T_ad"] = float(t_op["value"])

        # Surcharges de paramètres génériques
        for key, val in scenario.get("parameter_overrides", {}).items():
            overrides[key] = float(val["value"])

        return overrides

    @staticmethod
    def _flatten(params_dict: dict) -> dict:
        """
        Aplatit le dictionnaire YAML imbriqué (catégorie → paramètre → {value, ...}).
        Convertit toutes les valeurs numériques en float.
        """
        flat = {}
        for category in params_dict.values():
            if not isinstance(category, dict):
                continue
            for key, val in category.items():
                if isinstance(val, dict) and "value" in val:
                    flat[key] = float(val["value"])
        return flat

    # ------------------------------------------------------------------
    # Accès
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> float:
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"Paramètre ADM1 '{name}' introuvable.")

    def get(self, name: str, default=None):
        """Accès sécurisé avec valeur par défaut."""
        return self.params.get(name, default)

    def __repr__(self) -> str:
        return f"<ADM1Parameters: {len(self.params)} paramètres chargés>"
"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: ADM1Parameters

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026


—  Reorganised by Margaux Bonal
Email: margaux.bonal@inrae.fr
Date: 04/2026


Loads ADM1 model parameters from the modular configuration:
  - configs/adm1_parameters.yaml       : intrinsic model parameters
  - configs/scenarios/scenarios.yaml   : active scenario and overrides
Simulation-specific parameters (T_op, q_ad, V_liq, etc.) can be overridden
by the active scenario.
"""

import yaml


class ADM1Parameters:
    """
    Loads and exposes ADM1 model parameters.

    Resolution order (decreasing priority):
        1. parameter_overrides of the active scenario
        2. T_op value of the active scenario
        3. adm1_parameters.yaml (central reference)
    """

    def __init__(
        self,
        params_file: str = "configs/adm1_parameters.yaml",
        scenarios_file: str = "configs/Scenario.yaml",
    ):
        self.params_file = params_file
        self.scenarios_file = scenarios_file
        self.params: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self):
        """Load the base parameters then apply scenario overrides."""
        base = self._load_base_params()
        overrides = self._load_scenario_overrides()
        self.params = {**base, **overrides}

    def _load_base_params(self) -> dict:
        """Load and flatten adm1_parameters.yaml."""
        with open(self.params_file, "r") as f:
            raw = yaml.safe_load(f)
        return self._flatten(raw)

    def _load_scenario_overrides(self) -> dict:
        """
        Read the active scenario in scenarios.yaml and return:
          - T_op of the scenario (if present)
          - all parameter_overrides of the scenario
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

        # Scenario-specific operating temperature
        t_op = scenario.get("T_op")
        if t_op:
            overrides["T_op"] = float(t_op["value"])
            # T_ad is an alias kept for compatibility
            overrides["T_ad"] = float(t_op["value"])

        # Generic parameter overrides
        for key, val in scenario.get("parameter_overrides", {}).items():
            overrides[key] = float(val["value"])

        return overrides

    @staticmethod
    def _flatten(params_dict: dict) -> dict:
        """
        Flatten the nested YAML dict (category → parameter → {value, ...}).
        Converts every numeric value to float.
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
    # Access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> float:
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"ADM1 parameter '{name}' not found.")

    def get(self, name: str, default=None):
        """Safe access with a default value."""
        return self.params.get(name, default)

    def __repr__(self) -> str:
        return f"<ADM1Parameters: {len(self.params)} parameters loaded>"
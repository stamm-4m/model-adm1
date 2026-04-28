"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: Influent

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026


—  Reorganised by Margaux Bonal
Email: margaux.bonal@inrae.fr
Date: 04/2026


Loads the influent configuration from:
  configs/influent/influent.yaml      (mode definitions)
  configs/scenarios/scenarios.yaml    (active mode selector)

Two modes:
  - dynamic  : time series read from a CSV file
  - constant : constant values defined in influent.yaml
"""

import numpy as np
import pandas as pd
import yaml
from src.acid_base import compute_required_strong_ion_for_pH
from src.parameters import ADM1Parameters


# ADM1 influent variables (CSV column order for the dynamic mode)
INFLUENT_VARS = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac",
    "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
    "X_xc", "X_ch", "X_pr", "X_li",
    "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
    "S_cation", "S_anion",
]


class Influent:
    """
    Manages the digester input data.

    Usage:
        inf = Influent()
        inf.get(step)       -> dict with keys "S_su_in", "S_aa_in", ...
        inf.get_time()      -> np.ndarray of time points (dynamic mode only)
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
    # Loading
    # ------------------------------------------------------------------

    def _load(self):
        """Read the configuration and load data according to the active mode."""
        mode_key = self._get_active_mode()

        with open(self.influent_file, "r") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(
                f"File {self.influent_file} is empty or malformed "
                f"(yaml.safe_load returned: {type(raw).__name__})."
            )

        if mode_key is None or mode_key not in raw:
            available = list(raw.keys())
            raise KeyError(
                f"Influent mode '{mode_key}' not found in "
                f"{self.influent_file}. Available: {available}"
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

            # Dummy time vector [0, 1] so solve_ivp has something to start with
            self._time = np.array([0.0, 1.0])

        else:
            raise ValueError(f"Unknown influent mode: '{self.mode}'")

    def _get_active_mode(self) -> str:
        """Read the active scenario and return the influent-mode key."""
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
    # Access
    # ------------------------------------------------------------------

    def get(self, step: int) -> dict:
        """
        Return a dict of influent variables at time step `step`.
        Keys are suffixed with '_in' (e.g. "S_su_in").
        """
        if self.mode == "dynamic":
            step = min(step, len(self._time) - 1)
            return {var + "_in": self._data[var][step] for var in INFLUENT_VARS}
        else:
            return {var + "_in": self._constant_values.get(var, 0.0) for var in INFLUENT_VARS}

    def get_time(self) -> np.ndarray:
        """Return the simulation time vector."""
        return self._time

    def __repr__(self) -> str:
        n = len(self._time) if self._time is not None else 0
        return f"<Influent mode='{self.mode}' n_steps={n}>"

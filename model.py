"""
ADM1 on stamm-model-core (backlog 15, step 3, 2026-11-06).

The science is unchanged: the ODEs, rates, inhibitions and gas transfer (src/reactor.py), the
acid-base equilibrium (src/acid_base.py), the influent (src/influent.py) and the hybrid hooks
(src/hybrid.py, src/registry.py, models/*.yaml). This class only tells stamm-model-core how the
previous main.py ran them:

  * the 29 integrated states (src/reactor.py DYNAMIC_STATE_NAMES); the 9 acid-base species are
    solved algebraically inside every RHS call                                      -> rhs()
  * the initial state: the initial set, the hybrid hooks wired, the acid-base re-projection
                                                                                   -> initial_state()
  * K_w, K_a_co2, K_a_IN recomputed from T_op by ADM1Reactor                       -> derived_parameters()
  * the influent read by the model (zero-order hold on its time axis, dynamic flow Q(t)), the
    solver restarted at every influent row ("piecewise")                            -> breakpoints()
  * which influent row is held over [t_(i-1), t_i] (simulation.yaml influent_hold, backlog 63):
      start   row i-1: a row holds until the next one (default; best match of the MATLAB reference)
      end     row i, the convention of the PyADM1 loop (the row at the end of the interval)
      legacy  model-adm1 before 2026-11-06: row i-1, except the solver evaluations at t_i
              exactly, which read row i (to reproduce earlier runs only)
    with end / start the row is fixed for the whole solver piece                    -> before_segment()
  * the acid-base species, pH, COD and biogas flows of every output row, computed after the run in
    the same order as before (the acid-base Newton starts from the previous solution) -> post_columns()
  * a negative concentration flags the run as numerically invalid                 -> result_warnings()

Authors:
    - Margaux Bonal           <margaux.bonal@inrae.fr>
    - David Camilo Corrales   <David-Camilo.Corrales-Munoz@inrae.fr>
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stamm_model import OdeModel

from initial_states import STATE_VARIABLES
from src.acid_base import compute_acid_base_equilibrium, compute_total_cod
from src.hybrid import apply_hybrid_config
from src.parameters import ADM1Parameters
from src.reactor import DYNAMIC_STATE_NAMES, FULL_STATE_NAMES, ADM1Reactor

# recomputed from T_op in ADM1Reactor.__init__ (van 't Hoff, BSM2 Table 3)
COMPUTED_PARAMETERS = ("K_w", "K_a_co2", "K_a_IN")
INFLUENT_HOLDS = ("start", "end", "legacy")
GAS_COLUMNS = ("q_gas", "q_gas_atm", "q_gas_norm_dry", "q_ch4_norm_dry")


class ADM1(OdeModel):
    states = list(DYNAMIC_STATE_NAMES)

    def __init__(self, influent, hybrid_cfg: dict | None = None, registry: dict | None = None,
                 piecewise: bool = True, progress_s: float = 30.0, influent_hold: str = "start"):
        """``influent``: src.influent.Influent of the scenario; ``hybrid_cfg``: the scenario's
        adm1_hybrid block; ``registry``: src.registry.load_registry(); ``piecewise``: restart the
        solver at every influent row (dynamic influent only, simulation.yaml solver.piecewise);
        ``influent_hold``: end | start | legacy (see the module docstring)."""
        super().__init__()
        if influent_hold not in INFLUENT_HOLDS:
            raise ValueError(f"influent_hold must be one of {INFLUENT_HOLDS}, got {influent_hold!r}")
        self.influent_hold = influent_hold
        self._row = None                                     # the row held over the current piece
        self.influent = influent
        self.time_data = np.asarray(influent.get_time(), dtype=float)
        if len(self.time_data) == 0:
            raise ValueError("Empty influent: no time points found.")
        # read once: the solver calls rhs thousands of times per simulated day
        self.cache = [influent.get(i) for i in range(len(self.time_data))]
        self.piecewise = bool(piecewise) and getattr(influent, "mode", "constant") == "dynamic"
        self.hybrid_cfg = hybrid_cfg or {}
        self.registry = registry or {}
        self.param = None
        self.reactor = None
        self.hybrid_summary = None
        self.y0_full = None
        self.progress_s = float(progress_s)
        self._t_max = float("-inf")
        self._wall0 = None
        self._last_print = 0.0

    # ------------------------------------------------------------------ before the run
    def initial_state(self, x0, p):
        # x0: the initial set as declared (initial_states.yaml; S_H_ion / S_anion derived from its pH)
        missing = [v for v in STATE_VARIABLES if v not in x0]
        if missing:
            raise KeyError(f"Missing variables in the initial state: {missing}")
        self.param = ADM1Parameters.from_values(p.as_dict())
        y0_full = np.asarray([float(x0[v]) for v in STATE_VARIABLES], dtype=float)
        self.reactor = ADM1Reactor(self.param, constants=None)        # recomputes K_w, K_a_co2, K_a_IN
        self.hybrid_summary = apply_hybrid_config(self.reactor, self.hybrid_cfg, self.registry)
        # re-project onto the acid-base equilibrium: consistent pH / HCO3- / CO2 / NH3 / NH4+ at t_start
        eq = compute_acid_base_equilibrium(self.reactor.unpack_state(y0_full), self.param)
        y0_full = self.reactor._apply_acid_base_to_state_vector(y0_full, eq)
        self.y0_full = np.array(y0_full, dtype=float)
        y0 = self.reactor.reduce_to_dynamic_state(y0_full)
        return {name: float(v) for name, v in zip(DYNAMIC_STATE_NAMES, y0)}

    def derived_parameters(self, p):
        return {k: self.param.params[k] for k in COMPUTED_PARAMETERS}

    def breakpoints(self, t_start, t_end):
        if not self.piecewise:
            return []
        t = self.time_data
        return [float(v) for v in t[(t > t_start) & (t < t_end)]]

    def before_segment(self, t_start, t_end):
        # piecewise integration: one solver piece = one influent interval; its row is fixed here
        if not self.piecewise or self.influent_hold == "legacy":
            self._row = None
            return
        t, n = self.time_data, len(self.cache)
        if self.influent_hold == "end":
            i = int(np.searchsorted(t, t_end, side="left"))          # first row at or after t_end
        else:
            i = int(np.searchsorted(t, t_start, side="right") - 1)   # last row at or before t_start
        self._row = self.cache[max(0, min(i, n - 1))]

    # ------------------------------------------------------------------ the equations
    def influent_at(self, t: float) -> dict:
        """Feed row in force at t: the row of the current piece, else by the hold convention."""
        if self._row is not None:
            return self._row
        if self.influent_hold == "end":
            step = int(np.searchsorted(self.time_data, t, side="left"))
        else:   # start, legacy: the last row at or before t (legacy: also within a piece)
            step = int(np.searchsorted(self.time_data, t, side="right") - 1)
        step = max(0, min(step, len(self.cache) - 1))
        return self.cache[step]

    def rhs(self, t, y, p, u):
        self.reactor.influent_state = self.influent_at(t)
        if "q_ad_in" in self.reactor.influent_state:                 # dynamic feed flow Q(t)
            self.param.params["q_ad"] = self.reactor.influent_state["q_ad_in"]
        dydt = self.reactor.ADM1_ODE(t, y.array)
        self._progress(t)
        return dydt

    def _progress(self, t):
        if self.progress_s <= 0:
            return
        now = time.time()
        if self._wall0 is None:
            self._wall0 = now
        self._t_max = max(self._t_max, float(t))
        if now - self._last_print >= self.progress_s:
            self._last_print = now
            print(f"  solver at t = {self._t_max:.3f} d | wall-clock {now - self._wall0:6.1f} s", flush=True)

    # ------------------------------------------------------------------ after the run
    def post_columns(self, c, p):
        """Acid-base species, pH, COD and biogas flows of every row, as the previous main.py: a first
        acid-base pass rebuilds the 38-state rows, a second pass and the COD on each row, then the
        gas flows on the whole table."""
        reactor, param = self.reactor, self.param
        dynamic = np.column_stack([np.asarray(c[n], dtype=float) for n in DYNAMIC_STATE_NAMES])
        rows = []
        for dynamic_row in dynamic:
            full = reactor.expand_dynamic_state(dynamic_row)
            state = reactor.unpack_state(full)
            rows.append(reactor._apply_acid_base_to_state_vector(full, compute_acid_base_equilibrium(state, param)))
        df = pd.DataFrame(rows, columns=FULL_STATE_NAMES)
        df.insert(0, "time", np.asarray(c["time"], dtype=float))
        acid_base_rows, cod_rows = [], []
        for _, row in df.iterrows():
            state = {name: float(row[name]) for name in FULL_STATE_NAMES}
            acid_base_rows.append(compute_acid_base_equilibrium(state, param))
            cod_rows.append(compute_total_cod(state, param))
        for extra in (pd.DataFrame(acid_base_rows), pd.DataFrame(cod_rows)):
            for col in extra.columns:
                df[col] = extra[col]
        gas = reactor.compute_gas_outputs(df)
        for col in GAS_COLUMNS:
            df[col] = np.asarray(gas[col], dtype=float)
        integrated = set(DYNAMIC_STATE_NAMES) | {"time"}
        self.table = df                                           # for the plots of main.py
        return {col: df[col].to_numpy(dtype=float) for col in df.columns if col not in integrated}

    def result_warnings(self, c, p):
        Y = np.array([np.asarray(c[n], dtype=float) for n in DYNAMIC_STATE_NAMES])
        if Y.size == 0:
            return []
        neg = float(np.min(Y))
        if neg < -1e-9:
            i, j = np.unravel_index(np.argmin(Y), Y.shape)
            return [f"negative concentration {DYNAMIC_STATE_NAMES[i]} = {neg:.3e} at t = {c['time'][j]:.3f} d: "
                    "integration numerically unstable, results NOT valid"]
        return []

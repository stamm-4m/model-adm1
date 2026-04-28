"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: Reactor

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026


—  Réorganisé par Margaux Bonal 
Email : margaux.bonal@inrae.fr
Date : 04/2026


Convention gaz utilisée dans ce fichier :
- S_gas_h2  : état gaz H2   [kgCOD.m^-3]
- S_gas_ch4 : état gaz CH4  [kgCOD.m^-3]
- S_gas_co2 : état gaz CO2  [kmolC.m^-3]

Les pressions partielles sont toujours calculées à partir de ces états :
- p_gas_h2, p_gas_ch4, p_gas_co2 [bar]
"""
"""
ADM1 Reactor Simulation (adapted from PyADM1).
Corrections apportées dans cette version :
  BUG 1 — compute_gas_transfer : Rho_T_10 utilisait S_IC au lieu de S_co2
           → surestime le dégazage CO2, vide S_IC trop vite, fausse alcalinité et pH
  BUG 2 — mass_balances / ADM1_ODE : les variables DAE (S_H_ion, S_hco3_ion, S_nh3…)
           étaient figées à leurs valeurs initiales car compute_acid_base_equilibrium
           n'était jamais appelé → pH et équilibres acido-basiques artificiels
  BUG 3 — s_12 dans le bilan carbone : terme de consommation de C_IC manquant
           pour la méthanogenèse hydrogénotrophe (4H2 + CO2 → CH4 + 2H2O)

"""

import numpy as np

from src.acid_base import compute_acid_base_equilibrium, compute_total_cod
from initial_states import STATE_VARIABLES


FULL_STATE_NAMES = list(STATE_VARIABLES)
ALGEBRAIC_STATE_NAMES = [
    "S_H_ion",
    "S_va_ion",
    "S_bu_ion",
    "S_pro_ion",
    "S_ac_ion",
    "S_hco3_ion",
    "S_co2",
    "S_nh3",
    "S_nh4_ion",
]
DYNAMIC_STATE_NAMES = [name for name in FULL_STATE_NAMES if name not in ALGEBRAIC_STATE_NAMES]
DYNAMIC_STATE_INDICES = [FULL_STATE_NAMES.index(name) for name in DYNAMIC_STATE_NAMES]


class ADM1Reactor:

    def __init__(self, param, constants=None):
        self.param = param

        self.K_H_co2 = 0.035 * np.exp((-19410 / (100 * param.R)) * (1 / param.T_base - 1 / param.T_op))
        self.K_H_ch4 = 0.0014 * np.exp((-14240 / (100 * param.R)) * (1 / param.T_base - 1 / param.T_op))
        self.K_H_h2 = 7.8e-4 * np.exp((-4180 / (100 * param.R)) * (1 / param.T_base - 1 / param.T_op))
        self.p_gas_h2o = 0.0313 * np.exp(5290 * (1 / param.T_base - 1 / param.T_op))

        self.K_pH_aa = 10 ** (-(param.pH_LL_aa + param.pH_UL_aa) / 2)
        self.nn_aa = 3.0 / (param.pH_UL_aa - param.pH_LL_aa)
        self.K_pH_ac = 10 ** (-(param.pH_LL_ac + param.pH_UL_ac) / 2)
        self.n_ac = 3.0 / (param.pH_UL_ac - param.pH_LL_ac)
        self.K_pH_h2 = 10 ** (-(param.pH_LL_h2 + param.pH_UL_h2) / 2)
        self.n_h2 = 3.0 / (param.pH_UL_h2 - param.pH_LL_h2)

        self._state_template = np.zeros(len(FULL_STATE_NAMES), dtype=float)

    def ADM1_ODE(self, t, state_vector):
        input_is_reduced = len(state_vector) == len(DYNAMIC_STATE_NAMES)
        full_state_vector = (
            self.expand_dynamic_state(state_vector)
            if input_is_reduced
            else np.asarray(state_vector, dtype=float)
        )

        state = self.unpack_state(full_state_vector)

        # BUG 2: recompute acid-base algebraic states at every ODE call.
        eq = compute_acid_base_equilibrium(state, self.param)
        state.update(eq)
        full_state_vector = self._apply_acid_base_to_state_vector(full_state_vector, eq)

        inhib = self.compute_inhibitions(state)
        rho = self.compute_biochemical_rates(state, inhib)
        gas = self.compute_gas_transfer(state)

        full_derivatives = self.mass_balances(state, rho, gas)
        derivatives = (
            self.reduce_to_dynamic_state(full_derivatives)
            if input_is_reduced
            else full_derivatives
        )

        self._last_rho = rho
        self._last_inhib = inhib
        self._last_gas = gas
        self._last_state = self.unpack_state(full_state_vector)

        return derivatives

    def get_process_summary(self) -> dict:
        if not hasattr(self, "_last_rho"):
            return {}

        rho = self._last_rho
        inhib = self._last_inhib
        gas = self._last_gas
        state = self._last_state

        desintegration = rho["Rho_1"]
        hydrolyse = rho["Rho_2"] + rho["Rho_3"] + rho["Rho_4"]
        acidogenese = rho["Rho_5"] + rho["Rho_6"]
        acetogenese = rho["Rho_7"] + rho["Rho_8"] + rho["Rho_9"] + rho["Rho_10"]
        methanogenese = rho["Rho_11"] + rho["Rho_12"]
        decay = sum(rho[f"Rho_{i}"] for i in range(13, 20))

        pH = -np.log10(max(state["S_H_ion"], 1e-14))
        I_pH_aa = inhib["I_5"]
        I_nh3 = inhib["I_nh3"]
        I_h2_fa = inhib["I_7"] / max(inhib["I_5"], 1e-12) if inhib["I_5"] > 0 else 0.0
        p_h2, p_ch4, p_co2 = self.gas_state_to_partial_pressures(
            S_gas_h2=state["S_gas_h2"],
            S_gas_ch4=state["S_gas_ch4"],
            S_gas_co2=state["S_gas_co2"],
        )
        cod = compute_total_cod(state, self.param)

        return {
            "desintegration": desintegration,
            "hydrolyse": hydrolyse,
            "acidogenese": acidogenese,
            "acetogenese": acetogenese,
            "methanogenese": methanogenese,
            "decay": decay,
            "q_gas": gas["q_gas"],
            "pH": pH,
            "I_pH_aa": I_pH_aa,
            "I_nh3": I_nh3,
            "I_h2_fa": I_h2_fa,
            "p_gas_h2_bar": float(np.asarray(p_h2)),
            "p_gas_ch4_bar": float(np.asarray(p_ch4)),
            "p_gas_co2_bar": float(np.asarray(p_co2)),
            "COD_liquid_total": cod["COD_liquid_total"],
            "COD_total_inventory": cod["COD_total_inventory"],
        }

    def unpack_state(self, y) -> dict:
        return dict(zip(FULL_STATE_NAMES, y))

    def reduce_to_dynamic_state(self, full_state_vector):
        full_state = np.asarray(full_state_vector, dtype=float)
        if len(full_state) != len(FULL_STATE_NAMES):
            raise ValueError(
                f"Expected {len(FULL_STATE_NAMES)} full states, got {len(full_state)}."
            )
        self._state_template = full_state.copy()
        return full_state[DYNAMIC_STATE_INDICES]

    def expand_dynamic_state(self, dynamic_state_vector):
        state = np.asarray(dynamic_state_vector, dtype=float)
        if len(state) == len(FULL_STATE_NAMES):
            self._state_template = state.copy()
            return state.copy()
        if len(state) != len(DYNAMIC_STATE_NAMES):
            raise ValueError(
                f"Expected {len(DYNAMIC_STATE_NAMES)} dynamic states, got {len(state)}."
            )
        full_state = self._state_template.copy()
        full_state[DYNAMIC_STATE_INDICES] = state
        return full_state

    def _apply_acid_base_to_state_vector(self, state_vector, acid_base):
        full_state = self.expand_dynamic_state(state_vector)
        state = self.unpack_state(full_state)
        state.update(acid_base)
        updated = np.array([state[name] for name in FULL_STATE_NAMES], dtype=float)
        self._state_template = updated.copy()
        return updated

    def gas_state_to_partial_pressures(self, S_gas_h2, S_gas_ch4, S_gas_co2):
        p = self.param

        S_gas_h2 = np.asarray(S_gas_h2, dtype=float)
        S_gas_ch4 = np.asarray(S_gas_ch4, dtype=float)
        S_gas_co2 = np.asarray(S_gas_co2, dtype=float)

        p_gas_h2 = S_gas_h2 * p.R * p.T_op / 16.0
        p_gas_ch4 = S_gas_ch4 * p.R * p.T_op / 64.0
        p_gas_co2 = S_gas_co2 * p.R * p.T_op

        return p_gas_h2, p_gas_ch4, p_gas_co2

    def compute_gas_flow_rate(self, p_gas_h2, p_gas_ch4, p_gas_co2):
        p = self.param

        p_gas_h2 = np.asarray(p_gas_h2, dtype=float)
        p_gas_ch4 = np.asarray(p_gas_ch4, dtype=float)
        p_gas_co2 = np.asarray(p_gas_co2, dtype=float)

        p_gas_total = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o
        return np.maximum(0.0, p.k_p * (p_gas_total - p.p_atm))

    def compute_inhibitions(self, state) -> dict:
        p = self.param

        S_H_ion = state["S_H_ion"]
        S_IN = max(state["S_IN"], 1e-16)
        S_h2 = state["S_h2"]
        S_nh3 = state["S_nh3"]

        I_pH_aa = (self.K_pH_aa ** self.nn_aa) / (S_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa)
        I_pH_ac = (self.K_pH_ac ** self.n_ac) / (S_H_ion ** self.n_ac + self.K_pH_ac ** self.n_ac)
        I_pH_h2 = (self.K_pH_h2 ** self.n_h2) / (S_H_ion ** self.n_h2 + self.K_pH_h2 ** self.n_h2)
        I_IN_lim = 1 / (1 + (p.K_S_IN / S_IN))
        I_h2_fa = 1 / (1 + (S_h2 / p.K_I_h2_fa))
        I_h2_c4 = 1 / (1 + (S_h2 / p.K_I_h2_c4))
        I_h2_pro = 1 / (1 + (S_h2 / p.K_I_h2_pro))
        I_nh3 = 1 / (1 + (S_nh3 / p.K_I_nh3))

        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim

        return {
            "I_5": I_5,
            "I_6": I_6,
            "I_7": I_7,
            "I_8": I_8,
            "I_9": I_9,
            "I_10": I_10,
            "I_11": I_11,
            "I_12": I_12,
            "I_nh3": I_nh3,
        }

    def compute_biochemical_rates(self, state, inhib) -> dict:
        p = self.param

        S_su = state["S_su"]
        S_aa = state["S_aa"]
        S_fa = state["S_fa"]
        S_va = state["S_va"]
        S_bu = state["S_bu"]
        S_pro = state["S_pro"]
        S_ac = state["S_ac"]
        S_h2 = state["S_h2"]
        X_xc = state["X_xc"]
        X_ch = state["X_ch"]
        X_pr = state["X_pr"]
        X_li = state["X_li"]
        X_su = state["X_su"]
        X_aa = state["X_aa"]
        X_fa = state["X_fa"]
        X_c4 = state["X_c4"]
        X_pro = state["X_pro"]
        X_ac = state["X_ac"]
        X_h2 = state["X_h2"]

        I_5 = inhib["I_5"]
        I_6 = inhib["I_6"]
        I_7 = inhib["I_7"]
        I_8 = inhib["I_8"]
        I_9 = inhib["I_9"]
        I_10 = inhib["I_10"]
        I_11 = inhib["I_11"]
        I_12 = inhib["I_12"]

        Rho_1 = p.k_dis * X_xc
        Rho_2 = p.k_hyd_ch * X_ch
        Rho_3 = p.k_hyd_pr * X_pr
        Rho_4 = p.k_hyd_li * X_li
        Rho_5 = p.k_m_su * S_su / (p.K_S_su + S_su) * X_su * I_5
        Rho_6 = p.k_m_aa * S_aa / (p.K_S_aa + S_aa) * X_aa * I_6
        Rho_7 = p.k_m_fa * S_fa / (p.K_S_fa + S_fa) * X_fa * I_7
        Rho_8 = p.k_m_c4 * S_va / (p.K_S_c4 + S_va) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
        Rho_9 = p.k_m_c4 * S_bu / (p.K_S_c4 + S_bu) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        Rho_10 = p.k_m_pro * S_pro / (p.K_S_pro + S_pro) * X_pro * I_10
        Rho_11 = p.k_m_ac * S_ac / (p.K_S_ac + S_ac) * X_ac * I_11
        Rho_12 = p.k_m_h2 * S_h2 / (p.K_S_h2 + S_h2) * X_h2 * I_12
        Rho_13 = p.k_dec_X_su * X_su
        Rho_14 = p.k_dec_X_aa * X_aa
        Rho_15 = p.k_dec_X_fa * X_fa
        Rho_16 = p.k_dec_X_c4 * X_c4
        Rho_17 = p.k_dec_X_pro * X_pro
        Rho_18 = p.k_dec_X_ac * X_ac
        Rho_19 = p.k_dec_X_h2 * X_h2

        return {
            "Rho_1": Rho_1,
            "Rho_2": Rho_2,
            "Rho_3": Rho_3,
            "Rho_4": Rho_4,
            "Rho_5": Rho_5,
            "Rho_6": Rho_6,
            "Rho_7": Rho_7,
            "Rho_8": Rho_8,
            "Rho_9": Rho_9,
            "Rho_10": Rho_10,
            "Rho_11": Rho_11,
            "Rho_12": Rho_12,
            "Rho_13": Rho_13,
            "Rho_14": Rho_14,
            "Rho_15": Rho_15,
            "Rho_16": Rho_16,
            "Rho_17": Rho_17,
            "Rho_18": Rho_18,
            "Rho_19": Rho_19,
        }

    def compute_gas_transfer(self, state) -> dict:
        p = self.param

        S_h2 = state["S_h2"]
        S_ch4 = state["S_ch4"]
        S_co2 = state["S_co2"]
        S_gas_h2 = state["S_gas_h2"]
        S_gas_ch4 = state["S_gas_ch4"]
        S_gas_co2 = state["S_gas_co2"]

        p_gas_h2 = S_gas_h2 * p.R * p.T_op / 16
        p_gas_ch4 = S_gas_ch4 * p.R * p.T_op / 64
        p_gas_co2 = S_gas_co2 * p.R * p.T_op

        p_gas_total = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o
        q_gas = max(0.0, p.k_p * (p_gas_total - p.p_atm))

        Rho_T_8 = p.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)
        Rho_T_9 = p.k_L_a * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4)

        # BUG 1: use dissolved free CO2, not total inorganic carbon.
        Rho_T_10 = p.k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2)

        return {
            "q_gas": q_gas,
            "Rho_T_8": Rho_T_8,
            "Rho_T_9": Rho_T_9,
            "Rho_T_10": Rho_T_10,
        }

    def mass_balances(self, state, rho, gas) -> list:
        p = self.param
        infl = self.influent_state

        S_su_in = infl["S_su_in"]
        S_aa_in = infl["S_aa_in"]
        S_fa_in = infl["S_fa_in"]
        S_va_in = infl["S_va_in"]
        S_bu_in = infl["S_bu_in"]
        S_pro_in = infl["S_pro_in"]
        S_ac_in = infl["S_ac_in"]
        S_h2_in = infl["S_h2_in"]
        S_ch4_in = infl["S_ch4_in"]
        S_IC_in = infl["S_IC_in"]
        S_IN_in = infl["S_IN_in"]
        S_I_in = infl["S_I_in"]
        X_xc_in = infl["X_xc_in"]
        X_ch_in = infl["X_ch_in"]
        X_pr_in = infl["X_pr_in"]
        X_li_in = infl["X_li_in"]
        X_su_in = infl["X_su_in"]
        X_aa_in = infl["X_aa_in"]
        X_fa_in = infl["X_fa_in"]
        X_c4_in = infl["X_c4_in"]
        X_pro_in = infl["X_pro_in"]
        X_ac_in = infl["X_ac_in"]
        X_h2_in = infl["X_h2_in"]
        X_I_in = infl["X_I_in"]
        S_cation_in = infl["S_cation_in"]
        S_anion_in = infl["S_anion_in"]

        q_gas = gas["q_gas"]
        Rho_T_8 = gas["Rho_T_8"]
        Rho_T_9 = gas["Rho_T_9"]
        Rho_T_10 = gas["Rho_T_10"]

        S_su = state["S_su"]
        S_aa = state["S_aa"]
        S_fa = state["S_fa"]
        S_va = state["S_va"]
        S_bu = state["S_bu"]
        S_pro = state["S_pro"]
        S_ac = state["S_ac"]
        S_h2 = state["S_h2"]
        S_ch4 = state["S_ch4"]
        S_IC = state["S_IC"]
        S_IN = state["S_IN"]
        S_I = state["S_I"]
        X_xc = state["X_xc"]
        X_ch = state["X_ch"]
        X_pr = state["X_pr"]
        X_li = state["X_li"]
        X_su = state["X_su"]
        X_aa = state["X_aa"]
        X_fa = state["X_fa"]
        X_c4 = state["X_c4"]
        X_pro = state["X_pro"]
        X_ac = state["X_ac"]
        X_h2 = state["X_h2"]
        X_I = state["X_I"]
        S_cation = state["S_cation"]
        S_anion = state["S_anion"]
        S_gas_h2 = state["S_gas_h2"]
        S_gas_ch4 = state["S_gas_ch4"]
        S_gas_co2 = state["S_gas_co2"]

        Rho_1 = rho["Rho_1"]
        Rho_2 = rho["Rho_2"]
        Rho_3 = rho["Rho_3"]
        Rho_4 = rho["Rho_4"]
        Rho_5 = rho["Rho_5"]
        Rho_6 = rho["Rho_6"]
        Rho_7 = rho["Rho_7"]
        Rho_8 = rho["Rho_8"]
        Rho_9 = rho["Rho_9"]
        Rho_10 = rho["Rho_10"]
        Rho_11 = rho["Rho_11"]
        Rho_12 = rho["Rho_12"]
        Rho_13 = rho["Rho_13"]
        Rho_14 = rho["Rho_14"]
        Rho_15 = rho["Rho_15"]
        Rho_16 = rho["Rho_16"]
        Rho_17 = rho["Rho_17"]
        Rho_18 = rho["Rho_18"]
        Rho_19 = rho["Rho_19"]

        diff_S_su = p.q_ad / p.V_liq * (S_su_in - S_su) + Rho_2 + (1 - p.f_fa_li) * Rho_4 - Rho_5
        diff_S_aa = p.q_ad / p.V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6
        diff_S_fa = p.q_ad / p.V_liq * (S_fa_in - S_fa) + p.f_fa_li * Rho_4 - Rho_7
        diff_S_va = p.q_ad / p.V_liq * (S_va_in - S_va) + (1 - p.Y_aa) * p.f_va_aa * Rho_6 - Rho_8
        diff_S_bu = p.q_ad / p.V_liq * (S_bu_in - S_bu) + (1 - p.Y_su) * p.f_bu_su * Rho_5 + (1 - p.Y_aa) * p.f_bu_aa * Rho_6 - Rho_9
        diff_S_pro = (
            p.q_ad / p.V_liq * (S_pro_in - S_pro)
            + (1 - p.Y_su) * p.f_pro_su * Rho_5
            + (1 - p.Y_aa) * p.f_pro_aa * Rho_6
            + (1 - p.Y_c4) * 0.54 * Rho_8
            - Rho_10
        )
        diff_S_ac = (
            p.q_ad / p.V_liq * (S_ac_in - S_ac)
            + (1 - p.Y_su) * p.f_ac_su * Rho_5
            + (1 - p.Y_aa) * p.f_ac_aa * Rho_6
            + (1 - p.Y_fa) * 0.7 * Rho_7
            + (1 - p.Y_c4) * 0.31 * Rho_8
            + (1 - p.Y_c4) * 0.8 * Rho_9
            + (1 - p.Y_pro) * 0.57 * Rho_10
            - Rho_11
        )
        diff_S_h2 = (
            p.q_ad / p.V_liq * (S_h2_in - S_h2)
            + (1 - p.Y_su) * p.f_h2_su * Rho_5
            + (1 - p.Y_aa) * p.f_h2_aa * Rho_6
            + (1 - p.Y_fa) * 0.3 * Rho_7
            + (1 - p.Y_c4) * 0.15 * Rho_8
            + (1 - p.Y_c4) * 0.2 * Rho_9
            + (1 - p.Y_pro) * 0.43 * Rho_10
            - Rho_12
            - Rho_T_8
        )
        diff_S_ch4 = p.q_ad / p.V_liq * (S_ch4_in - S_ch4) + (1 - p.Y_ac) * Rho_11 + (1 - p.Y_h2) * Rho_12 - Rho_T_9

        s_1 = -p.C_xc + p.f_sI_xc * p.C_sI + p.f_ch_xc * p.C_ch + p.f_pr_xc * p.C_pr + p.f_li_xc * p.C_li + p.f_xI_xc * p.C_xI
        s_2 = -p.C_ch + p.C_su
        s_3 = -p.C_pr + p.C_aa
        s_4 = -p.C_li + (1 - p.f_fa_li) * p.C_su + p.f_fa_li * p.C_fa
        s_5 = -p.C_su + (1 - p.Y_su) * (p.f_bu_su * p.C_bu + p.f_pro_su * p.C_pro + p.f_ac_su * p.C_ac) + p.Y_su * p.C_bac
        s_6 = -p.C_aa + (1 - p.Y_aa) * (p.f_va_aa * p.C_va + p.f_bu_aa * p.C_bu + p.f_pro_aa * p.C_pro + p.f_ac_aa * p.C_ac) + p.Y_aa * p.C_bac
        s_7 = -p.C_fa + (1 - p.Y_fa) * 0.7 * p.C_ac + p.Y_fa * p.C_bac
        s_8 = -p.C_va + (1 - p.Y_c4) * 0.54 * p.C_pro + (1 - p.Y_c4) * 0.31 * p.C_ac + p.Y_c4 * p.C_bac
        s_9 = -p.C_bu + (1 - p.Y_c4) * 0.8 * p.C_ac + p.Y_c4 * p.C_bac
        s_10 = -p.C_pro + (1 - p.Y_pro) * 0.57 * p.C_ac + p.Y_pro * p.C_bac
        s_11 = -p.C_ac + (1 - p.Y_ac) * p.C_ch4 + p.Y_ac * p.C_bac

        # BUG 3: hydrogenotrophic methanogenesis must remove inorganic carbon.
        # H2 carries no carbon, so the net sink from S_IC is the carbon ending up
        # in CH4 and newly formed biomass.
        s_12 = (1 - p.Y_h2) * p.C_ch4 + p.Y_h2 * p.C_bac

        s_13 = -p.C_bac + p.C_xc

        Sigma = (
            s_1 * Rho_1
            + s_2 * Rho_2
            + s_3 * Rho_3
            + s_4 * Rho_4
            + s_5 * Rho_5
            + s_6 * Rho_6
            + s_7 * Rho_7
            + s_8 * Rho_8
            + s_9 * Rho_9
            + s_10 * Rho_10
            + s_11 * Rho_11
            + s_12 * Rho_12
            + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )

        diff_S_IC = p.q_ad / p.V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
        diff_S_IN = (
            p.q_ad / p.V_liq * (S_IN_in - S_IN)
            + (p.N_xc - p.f_xI_xc * p.N_I - p.f_sI_xc * p.N_I - p.f_pr_xc * p.N_aa) * Rho_1
            - p.Y_su * p.N_bac * Rho_5
            + (p.N_aa - p.Y_aa * p.N_bac) * Rho_6
            - p.Y_fa * p.N_bac * Rho_7
            - p.Y_c4 * p.N_bac * Rho_8
            - p.Y_c4 * p.N_bac * Rho_9
            - p.Y_pro * p.N_bac * Rho_10
            - p.Y_ac * p.N_bac * Rho_11
            - p.Y_h2 * p.N_bac * Rho_12
            + (p.N_bac - p.N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )
        diff_S_I = p.q_ad / p.V_liq * (S_I_in - S_I) + p.f_sI_xc * Rho_1

        diff_X_xc = p.q_ad / p.V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19
        diff_X_ch = p.q_ad / p.V_liq * (X_ch_in - X_ch) + p.f_ch_xc * Rho_1 - Rho_2
        diff_X_pr = p.q_ad / p.V_liq * (X_pr_in - X_pr) + p.f_pr_xc * Rho_1 - Rho_3
        diff_X_li = p.q_ad / p.V_liq * (X_li_in - X_li) + p.f_li_xc * Rho_1 - Rho_4
        diff_X_su = p.q_ad / p.V_liq * (X_su_in - X_su) + p.Y_su * Rho_5 - Rho_13
        diff_X_aa = p.q_ad / p.V_liq * (X_aa_in - X_aa) + p.Y_aa * Rho_6 - Rho_14
        diff_X_fa = p.q_ad / p.V_liq * (X_fa_in - X_fa) + p.Y_fa * Rho_7 - Rho_15
        diff_X_c4 = p.q_ad / p.V_liq * (X_c4_in - X_c4) + p.Y_c4 * (Rho_8 + Rho_9) - Rho_16
        diff_X_pro = p.q_ad / p.V_liq * (X_pro_in - X_pro) + p.Y_pro * Rho_10 - Rho_17
        diff_X_ac = p.q_ad / p.V_liq * (X_ac_in - X_ac) + p.Y_ac * Rho_11 - Rho_18
        diff_X_h2 = p.q_ad / p.V_liq * (X_h2_in - X_h2) + p.Y_h2 * Rho_12 - Rho_19
        diff_X_I = p.q_ad / p.V_liq * (X_I_in - X_I) + p.f_xI_xc * Rho_1

        diff_S_cation = p.q_ad / p.V_liq * (S_cation_in - S_cation)
        diff_S_anion = p.q_ad / p.V_liq * (S_anion_in - S_anion)

        diff_S_H_ion = 0.0
        diff_S_va_ion = 0.0
        diff_S_bu_ion = 0.0
        diff_S_pro_ion = 0.0
        diff_S_ac_ion = 0.0
        diff_S_hco3_ion = 0.0
        diff_S_co2 = 0.0
        diff_S_nh3 = 0.0
        diff_S_nh4_ion = 0.0

        diff_S_gas_h2 = (-q_gas / p.V_gas * S_gas_h2) + (Rho_T_8 * p.V_liq / p.V_gas)
        diff_S_gas_ch4 = (-q_gas / p.V_gas * S_gas_ch4) + (Rho_T_9 * p.V_liq / p.V_gas)
        diff_S_gas_co2 = (-q_gas / p.V_gas * S_gas_co2) + (Rho_T_10 * p.V_liq / p.V_gas)

        return [
            diff_S_su,
            diff_S_aa,
            diff_S_fa,
            diff_S_va,
            diff_S_bu,
            diff_S_pro,
            diff_S_ac,
            diff_S_h2,
            diff_S_ch4,
            diff_S_IC,
            diff_S_IN,
            diff_S_I,
            diff_X_xc,
            diff_X_ch,
            diff_X_pr,
            diff_X_li,
            diff_X_su,
            diff_X_aa,
            diff_X_fa,
            diff_X_c4,
            diff_X_pro,
            diff_X_ac,
            diff_X_h2,
            diff_X_I,
            diff_S_cation,
            diff_S_anion,
            diff_S_H_ion,
            diff_S_va_ion,
            diff_S_bu_ion,
            diff_S_pro_ion,
            diff_S_ac_ion,
            diff_S_hco3_ion,
            diff_S_co2,
            diff_S_nh3,
            diff_S_nh4_ion,
            diff_S_gas_h2,
            diff_S_gas_ch4,
            diff_S_gas_co2,
        ]

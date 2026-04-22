"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: Reactor

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026

"""

import numpy as np

class ADM1Reactor:

    def __init__(self, param, constants):

        self.param = param
        self.constants = constants
        # compute constants once
        self.K_H_co2 = 0.035 * np.exp((-19410 / (100 * param.R)) * (1 / param.T_base - 1 / param.T_ad))
        self.K_H_ch4 = 0.0014 * np.exp((-14240 / (100 * param.R)) * (1 / param.T_base - 1 / param.T_ad))
        self.K_H_h2 = 7.8e-4 * np.exp((-4180 / (100 * param.R)) * (1 / param.T_base - 1 / param.T_ad))
        self.p_gas_h2o = 0.0313 * np.exp(5290 * (1 / param.T_base - 1 / param.T_ad))

        self.K_pH_aa = 10 ** (-(param.pH_LL_aa + param.pH_UL_aa) / 2)
        self.nn_aa = 3.0 / (param.pH_UL_aa - param.pH_LL_aa)
        self.K_pH_ac = 10 ** (-(param.pH_LL_ac + param.pH_UL_ac) / 2)
        self.n_ac = 3.0 / (param.pH_UL_ac - param.pH_LL_ac)
        self.K_pH_h2 = 10 ** (-(param.pH_LL_h2 + param.pH_UL_h2) / 2)
        self.n_h2 = 3.0 / (param.pH_UL_h2 - param.pH_LL_h2)

    def ADM1_ODE(self, t, state_vector):

        state = self.unpack_state(state_vector)

        inhib = self.compute_inhibitions(state)

        rho = self.compute_biochemical_rates(state, inhib)

        gas = self.compute_gas_transfer(state)

        derivatives = self.mass_balances(state, rho, gas)

        return derivatives
    
    def unpack_state(self, y):


        (S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4,
        S_IC, S_IN, S_I,
        X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I,
        S_cation, S_anion,
        S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion,
        S_co2, S_nh3, S_nh4_ion,
        S_gas_h2, S_gas_ch4, S_gas_co2) = y

        return {
            "S_su": S_su,
            "S_aa": S_aa,
            "S_fa": S_fa,
            "S_va": S_va,
            "S_bu": S_bu,
            "S_pro": S_pro,
            "S_ac": S_ac,
            "S_h2": S_h2,
            "S_ch4": S_ch4,
            "S_IC": S_IC,
            "S_IN": S_IN,
            "S_I": S_I,
            "X_xc": X_xc,
            "X_ch": X_ch,
            "X_pr": X_pr,
            "X_li": X_li,
            "X_su": X_su,
            "X_aa": X_aa,
            "X_fa": X_fa,
            "X_c4": X_c4,
            "X_pro": X_pro,
            "X_ac": X_ac,
            "X_h2": X_h2,
            "X_I": X_I,
            "S_cation": S_cation,
            "S_anion": S_anion,
            "S_H_ion": S_H_ion,
            "S_va_ion": S_va_ion,
            "S_bu_ion": S_bu_ion,
            "S_pro_ion": S_pro_ion,
            "S_ac_ion": S_ac_ion,
            "S_hco3_ion": S_hco3_ion,
            "S_co2": S_co2,
            "S_nh3": S_nh3,
            "S_nh4_ion": S_nh4_ion,
            "S_gas_h2": S_gas_h2,
            "S_gas_ch4": S_gas_ch4,
            "S_gas_co2": S_gas_co2
        } 

    def compute_inhibitions(self, state):

        p = self.param

        S_H_ion = state["S_H_ion"]
        S_IN = state["S_IN"]
        S_h2 = state["S_h2"]
        S_nh3 = state["S_nh3"]

        # ------------------------------------------------
        # MOVE ORIGINAL INHIBITION EQUATIONS HERE
        # ------------------------------------------------

        I_pH_aa = (self.K_pH_aa ** self.nn_aa) / (S_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa)
        I_pH_ac = (self.K_pH_ac ** self.n_ac) / (S_H_ion ** self.n_ac + self.K_pH_ac ** self.n_ac)
        I_pH_h2 = (self.K_pH_h2 ** self.n_h2) / (S_H_ion ** self.n_h2 + self.K_pH_h2 ** self.n_h2)
        I_IN_lim =  (1 / (1 + (p.K_S_IN / S_IN)))
        I_h2_fa =  (1 / (1 + (S_h2 / p.K_I_h2_fa)))
        I_h2_c4 =  (1 / (1 + (S_h2 / p.K_I_h2_c4)))
        I_h2_pro =  (1 / (1 + (S_h2 / p.K_I_h2_pro)))
        I_nh3 =  1 / (1 + (S_nh3 / p.K_I_nh3))

        I_5 =  (I_pH_aa * I_IN_lim)
        I_6 = I_5
        I_7 =  (I_pH_aa * I_IN_lim * I_h2_fa)
        I_8 =  (I_pH_aa * I_IN_lim * I_h2_c4)
        I_9 = I_8
        I_10 =  (I_pH_aa * I_IN_lim * I_h2_pro)
        I_11 =  (I_pH_ac * I_IN_lim * I_nh3)
        I_12 =  (I_pH_h2 * I_IN_lim)

        return {
            "I_5": I_5,
            "I_6": I_6,
            "I_7": I_7,
            "I_8": I_8,
            "I_9": I_9,
            "I_10": I_10,
            "I_11": I_11,
            "I_12": I_12
        }

    def compute_biochemical_rates(self, state, inhib):

        p = self.param

        # access variables
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

        # inhibition terms
        I_5 = inhib["I_5"]
        I_6 = inhib["I_6"]
        I_7 = inhib["I_7"]
        I_8 = inhib["I_8"]
        I_9 = inhib["I_9"]
        I_10 = inhib["I_10"]
        I_11 = inhib["I_11"]
        I_12 = inhib["I_12"]

        # ------------------------------------------------
        # MOVE ORIGINAL RHO EQUATIONS HERE
        # ------------------------------------------------

        # biochemical process rates from Rosen et al (2006) BSM2 report
        Rho_1  =  (p.k_dis * X_xc)   # Disintegration
        Rho_2  =  (p.k_hyd_ch * X_ch)  # Hydrolysis of carbohydrates
        Rho_3  =  (p.k_hyd_pr * X_pr)  # Hydrolysis of proteins
        Rho_4  =  (p.k_hyd_li * X_li)  # Hydrolysis of lipids
        Rho_5  =  p.k_m_su * S_su / (p.K_S_su + S_su) * X_su * I_5  # Uptake of sugars
        Rho_6  =  (p.k_m_aa * (S_aa / (p.K_S_aa + S_aa)) * X_aa * I_6)  # Uptake of amino-acids
        Rho_7  =  (p.k_m_fa * (S_fa / (p.K_S_fa + S_fa)) * X_fa * I_7)  # Uptake of LCFA (long-chain fatty acids)
        Rho_8  =  (p.k_m_c4 * (S_va / (p.K_S_c4 + S_va )) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8)  # Uptake of valerate
        Rho_9  =  (p.k_m_c4 * (S_bu / (p.K_S_c4 + S_bu )) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9)  # Uptake of butyrate
        Rho_10 =  (p.k_m_pro * (S_pro / (p.K_S_pro + S_pro)) * X_pro * I_10)  # Uptake of propionate
        Rho_11 =  (p.k_m_ac * (S_ac / (p.K_S_ac + S_ac)) * X_ac * I_11)  # Uptake of acetate
        Rho_12 =  (p.k_m_h2 * (S_h2 / (p.K_S_h2 + S_h2)) * X_h2 * I_12)  # Uptake of hydrogen
        Rho_13 =  (p.k_dec_X_su * X_su)  # Decay of X_su
        Rho_14 =  (p.k_dec_X_aa * X_aa)  # Decay of X_aa
        Rho_15 =  (p.k_dec_X_fa * X_fa)  # Decay of X_fa
        Rho_16 =  (p.k_dec_X_c4 * X_c4)  # Decay of X_c4
        Rho_17 =  (p.k_dec_X_pro * X_pro)  # Decay of X_pro
        Rho_18 =  (p.k_dec_X_ac * X_ac)  # Decay of X_ac
        Rho_19 =  (p.k_dec_X_h2 * X_h2)  # Decay of X_h2

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
            "Rho_19": Rho_19
        }
    

    def compute_gas_transfer(self, state):

        p = self.param

        S_gas_h2 = state["S_gas_h2"]
        S_gas_ch4 = state["S_gas_ch4"]
        S_gas_co2 = state["S_gas_co2"]

        S_h2  = state["S_h2"]
        S_ch4 = state["S_ch4"]
        S_co2 = state["S_co2"]

        # ------------------------------------------------
        # MOVE ORIGINAL GAS EQUATIONS HERE
        # ------------------------------------------------

        # gas phase algebraic equations from Rosen et al (2006) BSM2 report
        p_gas_h2 =  (S_gas_h2 * p.R * p.T_op / 16)
        p_gas_ch4 =  (S_gas_ch4 * p.R * p.T_op / 64)
        p_gas_co2 =  (S_gas_co2 * p.R * p.T_op)


        p_gas=  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o)
        q_gas =  (p.k_p * (p_gas- p.p_atm))
        if q_gas < 0:    q_gas = 0

        q_ch4 = q_gas * (p_gas_ch4/p_gas) # methane flow

        # gas transfer rates from Rosen et al (2006) BSM2 report
        Rho_T_8 =  (p.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2))
        Rho_T_9 =  (p.k_L_a * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4))
        Rho_T_10 =  (p.k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2))

        return {
            "p_gas": p_gas,
            "q_gas": q_gas,
            "Rho_T_8": Rho_T_8,
            "Rho_T_9": Rho_T_9,
            "Rho_T_10": Rho_T_10
        }
            

    def mass_balances(self, state, rho, gas):

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

        # access rho
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

        # ------------------------------------------------
        # MOVE ORIGINAL DIFFERENTIAL EQUATIONS HERE
        # ------------------------------------------------

        ##differential equaitons from Rosen et al (2006) BSM2 report
        # differential equations 1 to 12 (soluble matter)
        diff_S_su = p.q_ad /p.V_liq * (S_su_in - S_su) + Rho_2 + (1 -p.f_fa_li) * Rho_4 - Rho_5  # eq1

        diff_S_aa = p.q_ad /p.V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6  # eq2

        diff_S_fa = p.q_ad /p.V_liq * (S_fa_in - S_fa) + (p.f_fa_li * Rho_4) - Rho_7  # eq3

        diff_S_va = p.q_ad /p.V_liq * (S_va_in - S_va) + (1 -p.Y_aa) *p.f_va_aa * Rho_6 - Rho_8  # eq4

        diff_S_bu = p.q_ad /p.V_liq * (S_bu_in - S_bu) + (1 -p.Y_su) *p.f_bu_su * Rho_5 + (1 -p.Y_aa) *p.f_bu_aa * Rho_6 - Rho_9  # eq5

        diff_S_pro = p.q_ad /p.V_liq * (S_pro_in - S_pro) + (1 -p.Y_su) *p.f_pro_su * Rho_5 + (1 -p.Y_aa) *p.f_pro_aa * Rho_6 + (1 -p.Y_c4) * 0.54 * Rho_8 - Rho_10  # eq6

        diff_S_ac = p.q_ad /p.V_liq * (S_ac_in - S_ac) + (1 -p.Y_su) *p.f_ac_su * Rho_5 + (1 -p.Y_aa) *p.f_ac_aa * Rho_6 + (1 -p.Y_fa) * 0.7 * Rho_7 + (1 -p.Y_c4) * 0.31 * Rho_8 + (1 -p.Y_c4) * 0.8 * Rho_9 + (1 -p.Y_pro) * 0.57 * Rho_10 - Rho_11  # eq7

        #diff_S_h2 is defined with DAE paralel equaitons

        diff_S_ch4 = p.q_ad /p.V_liq * (S_ch4_in - S_ch4) + (1 -p.Y_ac) * Rho_11 + (1 -p.Y_h2) * Rho_12 - Rho_T_9  # eq9


        ## eq10 start##
        s_1 =  (-1 *p.C_xc +p.f_sI_xc *p.C_sI +p.f_ch_xc *p.C_ch +p.f_pr_xc *p.C_pr +p.f_li_xc *p.C_li +p.f_xI_xc *p.C_xI) 
        s_2 =  (-1 *p.C_ch +p.C_su)
        s_3 =  (-1 *p.C_pr +p.C_aa)
        s_4 =  (-1 *p.C_li + (1 -p.f_fa_li) *p.C_su +p.f_fa_li *p.C_fa)
        s_5 =  (-1 *p.C_su + (1 -p.Y_su) * (p.f_bu_su *p.C_bu +p.f_pro_su *p.C_pro +p.f_ac_su *p.C_ac) +p.Y_su *p.C_bac)
        s_6 =  (-1 *p.C_aa + (1 -p.Y_aa) * (p.f_va_aa *p.C_va +p.f_bu_aa *p.C_bu +p.f_pro_aa *p.C_pro +p.f_ac_aa *p.C_ac) +p.Y_aa *p.C_bac)
        s_7 =  (-1 *p.C_fa + (1 -p.Y_fa) * 0.7 *p.C_ac +p.Y_fa *p.C_bac)
        s_8 =  (-1 *p.C_va + (1 -p.Y_c4) * 0.54 *p.C_pro + (1 -p.Y_c4) * 0.31 *p.C_ac +p.Y_c4 *p.C_bac)
        s_9 =  (-1 *p.C_bu + (1 -p.Y_c4) * 0.8 *p.C_ac +p.Y_c4 *p.C_bac)
        s_10 =  (-1 *p.C_pro + (1 -p.Y_pro) * 0.57 *p.C_ac +p.Y_pro *p.C_bac)
        s_11 =  (-1 *p.C_ac + (1 -p.Y_ac) *p.C_ch4 +p.Y_ac *p.C_bac)
        s_12 =  ((1 -p.Y_h2) *p.C_ch4 +p.Y_h2 *p.C_bac)
        s_13 =  (-1 *p.C_bac +p.C_xc) 

        Sigma =  (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))

        diff_S_IC = p.q_ad /p.V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
        ## eq10 end##


        
        diff_S_IN = p.q_ad /p.V_liq * (S_IN_in - S_IN) + (p.N_xc -p.f_xI_xc *p.N_I -p.f_sI_xc *p.N_I-p.f_pr_xc *p.N_aa) * Rho_1 -p.Y_su *p.N_bac * Rho_5 + (p.N_aa -p.Y_aa *p.N_bac) * Rho_6 -p.Y_fa *p.N_bac * Rho_7 -p.Y_c4 *p.N_bac * Rho_8 -p.Y_c4 *p.N_bac * Rho_9 -p.Y_pro *p.N_bac * Rho_10 -p.Y_ac *p.N_bac * Rho_11 -p.Y_h2 *p.N_bac * Rho_12 + (p.N_bac -p.N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19) # eq11 


        diff_S_I = p.q_ad /p.V_liq * (S_I_in - S_I) +p.f_sI_xc * Rho_1  # eq12


        # Differential equations 13 to 24 (particulate matter)
        diff_X_xc = p.q_ad /p.V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19  # eq13 

        diff_X_ch = p.q_ad /p.V_liq * (X_ch_in - X_ch) +p.f_ch_xc * Rho_1 - Rho_2  # eq14 

        diff_X_pr = p.q_ad /p.V_liq * (X_pr_in - X_pr) +p.f_pr_xc * Rho_1 - Rho_3  # eq15 

        diff_X_li = p.q_ad /p.V_liq * (X_li_in - X_li) +p.f_li_xc * Rho_1 - Rho_4  # eq16 

        diff_X_su = p.q_ad /p.V_liq * (X_su_in - X_su) +p.Y_su * Rho_5 - Rho_13  # eq17

        diff_X_aa = p.q_ad /p.V_liq * (X_aa_in - X_aa) +p.Y_aa * Rho_6 - Rho_14  # eq18

        diff_X_fa = p.q_ad /p.V_liq * (X_fa_in - X_fa) +p.Y_fa * Rho_7 - Rho_15  # eq19

        diff_X_c4 = p.q_ad /p.V_liq * (X_c4_in - X_c4) +p.Y_c4 * Rho_8 +p.Y_c4 * Rho_9 - Rho_16  # eq20

        diff_X_pro = p.q_ad /p.V_liq * (X_pro_in - X_pro) +p.Y_pro * Rho_10 - Rho_17  # eq21

        diff_X_ac = p.q_ad /p.V_liq * (X_ac_in - X_ac) +p.Y_ac * Rho_11 - Rho_18  # eq22

        diff_X_h2 = p.q_ad /p.V_liq * (X_h2_in - X_h2) +p.Y_h2 * Rho_12 - Rho_19  # eq23

        diff_X_I = p.q_ad /p.V_liq * (X_I_in - X_I) +p.f_xI_xc * Rho_1  # eq24 

        # Differential equations 25 and 26 (cations and anions)
        diff_S_cation = p.q_ad /p.V_liq * (S_cation_in - S_cation)  # eq25

        diff_S_anion = p.q_ad /p.V_liq * (S_anion_in - S_anion)  # eq26

        diff_S_h2 = 0

        # Differential equations 27 to 32 (ion states, only for ODE implementation)
        diff_S_va_ion = 0  # eq27

        diff_S_bu_ion = 0  # eq28

        diff_S_pro_ion = 0  # eq29

        diff_S_ac_ion = 0  # eq30

        diff_S_hco3_ion = 0  # eq31

        diff_S_nh3 = 0  # eq32

        # Gas phase equations: Differential equations 33 to 35
        diff_S_gas_h2 = (q_gas /p.V_gas * -1 * S_gas_h2) + (Rho_T_8 *p.V_liq /p.V_gas)  # eq33

        diff_S_gas_ch4 = (q_gas /p.V_gas * -1 * S_gas_ch4) + (Rho_T_9 *p.V_liq /p.V_gas)  # eq34

        diff_S_gas_co2 = (q_gas /p.V_gas * -1 * S_gas_co2) + (Rho_T_10 *p.V_liq /p.V_gas)  # eq35

        diff_S_H_ion = 0
        diff_S_co2 = 0
        diff_S_nh4_ion = 0 #to keep the output same length as input for ADM1_ODE funcion
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
            diff_S_gas_co2
        ]
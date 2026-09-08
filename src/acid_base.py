"""
ADM1 acid-base equilibrium module.

Goals:
- recompute pH, HCO3-, CO2, NH3, NH4+
- prevent S_IC from going negative or becoming inconsistent
- keep the charge balance (electroneutrality) consistent
"""

import numpy as np


COD_EQUIVALENTS = {
    "va": 208.0,   # kgCOD / kmol
    "bu": 160.0,
    "pro": 112.0,
    "ac": 64.0,
}


H_FLOOR = 1e-14
CONCENTRATION_FLOOR = 1e-16


def _safe_positive(value: float, floor: float) -> float:
    return value if value > floor else floor


def _vfa_cod_to_molar(concentration_kgcod_m3: float, species: str) -> float:
    """
    Convert an ADM1 VFA concentration from kgCOD.m^-3 to kmol.m^-3.

    Acidity constants are defined in M (= kmol.m^-3), so this conversion
    must be done before evaluating the charge balance.
    """
    return max(concentration_kgcod_m3, 0.0) / COD_EQUIVALENTS[species]


def kgcod_m3_to_mol_l(concentration_kgcod_m3: float, species: str) -> float:
    """
    Explicit ADM1 conversion: kgCOD.m^-3 -> mol.L^-1.

    Numerically mol.L^-1 = kmol.m^-3, so the value matches the molar
    conversion used in the charge balance.
    """
    return _vfa_cod_to_molar(concentration_kgcod_m3, species)


def _charge_balance(H: float, state: dict, param) -> tuple[float, dict]:
    """
    Compute the electroneutrality residual for a given H+ concentration.

    Total VFA concentrations are stored in the state in kgCOD.m^-3.
    For the charge balance they are converted to kmol.m^-3 (= M).
    """

    H = _safe_positive(H, H_FLOOR)

    S_va_tot = _vfa_cod_to_molar(state["S_va"], "va")
    S_bu_tot = _vfa_cod_to_molar(state["S_bu"], "bu")
    S_pro_tot = _vfa_cod_to_molar(state["S_pro"], "pro")
    S_ac_tot = _vfa_cod_to_molar(state["S_ac"], "ac")
    S_IC = _safe_positive(state["S_IC"], CONCENTRATION_FLOOR)
    S_IN = _safe_positive(state["S_IN"], CONCENTRATION_FLOOR)
    S_cation = max(state["S_cation"], 0.0)
    S_anion = max(state["S_anion"], 0.0)

    S_va_ion = param.K_a_va * S_va_tot / (param.K_a_va + H)
    S_bu_ion = param.K_a_bu * S_bu_tot / (param.K_a_bu + H)
    S_pro_ion = param.K_a_pro * S_pro_tot / (param.K_a_pro + H)
    S_ac_ion = param.K_a_ac * S_ac_tot / (param.K_a_ac + H)
    S_hco3_ion = param.K_a_co2 * S_IC / (param.K_a_co2 + H)
    S_nh3 = param.K_a_IN * S_IN / (param.K_a_IN + H)
    S_nh4_ion = max(S_IN - S_nh3, 0.0)
    S_oh_ion = param.K_w / H
    S_co2 = max(S_IC - S_hco3_ion, 0.0)

    charge_residual = (
        S_cation
        + S_nh4_ion
        + H
        - S_hco3_ion
        - S_ac_ion
        - S_pro_ion
        - S_bu_ion
        - S_va_ion
        - S_oh_ion
        - S_anion
    )

    species = {
        "S_va_ion": S_va_ion,
        "S_bu_ion": S_bu_ion,
        "S_pro_ion": S_pro_ion,
        "S_ac_ion": S_ac_ion,
        "S_hco3_ion": S_hco3_ion,
        "S_nh3": S_nh3,
        "S_nh4_ion": S_nh4_ion,
        "S_oh_ion": S_oh_ion,
        "S_co2": S_co2,
        "S_IC": S_IC,
        "S_IN": S_IN,
    }
    return charge_residual, species


def compute_required_strong_ion_for_pH(
    state: dict,
    param,
    target_pH: float,
    solve_for: str = "S_anion",
) -> float:
    """
    Compute the strong-ion concentration required to close electroneutrality
    at a target pH.

    Parameters
    ----------
    solve_for : {"S_anion", "S_cation"}
        Selects which strong ion is adjusted to satisfy the charge balance.
    """
    H = 10.0 ** (-float(target_pH))
    _, species = _charge_balance(H, state, param)

    negative_terms = (
        species["S_hco3_ion"]
        + species["S_ac_ion"]
        + species["S_pro_ion"]
        + species["S_bu_ion"]
        + species["S_va_ion"]
        + species["S_oh_ion"]
    )
    positive_terms = max(state["S_IN"], CONCENTRATION_FLOOR) - species["S_nh3"] + H

    if solve_for == "S_anion":
        return max(max(state["S_cation"], 0.0) + positive_terms - negative_terms, 0.0)
    if solve_for == "S_cation":
        return max(max(state["S_anion"], 0.0) - positive_terms + negative_terms, 0.0)
    raise ValueError("solve_for must be either 'S_anion' or 'S_cation'")


def compute_acid_base_equilibrium(state: dict, param, tol: float = 1e-12, max_iter: int = 100) -> dict:
    """
    Solve the ADM1 acid-base equilibrium from:
    S_IC, S_IN, total VFAs, cations, anions.

    Returns an updated dict containing:
    - S_H_ion
    - pH
    - S_va_ion
    - S_bu_ion
    - S_pro_ion
    - S_ac_ion
    - S_hco3_ion
    - S_co2
    - S_nh3
    - S_nh4_ion
    """

    H_guess = _safe_positive(state.get("S_H_ion", 1e-7), H_FLOOR)

    # Bracketed search over a wide pH range to avoid any unbounded Newton step.
    H_min = H_FLOOR
    H_max = 1.0
    log_grid = np.linspace(np.log10(H_min), np.log10(H_max), max_iter + 20)

    bracket = None
    previous_H = H_guess
    previous_residual, _ = _charge_balance(previous_H, state, param)

    candidate_grid = np.unique(np.concatenate(([H_guess], 10 ** log_grid)))

    for current_H in candidate_grid:
        if current_H == previous_H:
            continue
        current_residual, _ = _charge_balance(current_H, state, param)
        if previous_residual == 0.0:
            bracket = (previous_H, previous_H)
            break
        if previous_residual * current_residual <= 0.0:
            bracket = (previous_H, current_H)
            break
        previous_H = current_H
        previous_residual = current_residual

    if bracket is None:
        # No sign change found: fall back to the point with the smallest residual.
        candidates = candidate_grid
        H = min(candidates, key=lambda h: abs(_charge_balance(h, state, param)[0]))
    else:
        H_low, H_high = bracket
        H = 0.5 * (H_low + H_high)
        for _ in range(max_iter):
            H = 0.5 * (H_low + H_high)
            residual_mid, _ = _charge_balance(H, state, param)
            if abs(residual_mid) < tol:
                break
            residual_low, _ = _charge_balance(H_low, state, param)
            if residual_low * residual_mid <= 0.0:
                H_high = H
            else:
                H_low = H

    charge_residual, species = _charge_balance(H, state, param)

    return {
        "S_H_ion": H,
        "pH": -np.log10(H),
        "S_va_ion": species["S_va_ion"],
        "S_bu_ion": species["S_bu_ion"],
        "S_pro_ion": species["S_pro_ion"],
        "S_ac_ion": species["S_ac_ion"],
        "S_hco3_ion": species["S_hco3_ion"],
        "S_co2": species["S_co2"],
        "S_nh3": species["S_nh3"],
        "S_nh4_ion": species["S_nh4_ion"],
        "S_IC": species["S_IC"],
        "S_IN": species["S_IN"],
        "charge_residual": charge_residual,
    }


def compute_total_cod(state: dict, param=None) -> dict:
    """
    Compute total-COD diagnostics.

    All ADM1 organic variables are already in kgCOD.m^-3. We do NOT count
    S_IC, S_IN, ions, CO2 in the gas phase, H+, etc.
    """

    soluble_cod = (
        state["S_su"]
        + state["S_aa"]
        + state["S_fa"]
        + state["S_va"]
        + state["S_bu"]
        + state["S_pro"]
        + state["S_ac"]
        + state["S_h2"]
        + state["S_ch4"]
        + state["S_I"]
    )

    particulate_cod = (
        state["X_xc"]
        + state["X_ch"]
        + state["X_pr"]
        + state["X_li"]
        + state["X_su"]
        + state["X_aa"]
        + state["X_fa"]
        + state["X_c4"]
        + state["X_pro"]
        + state["X_ac"]
        + state["X_h2"]
        + state["X_I"]
    )

    gas_cod = state["S_gas_h2"] + state["S_gas_ch4"]

    total_liquid_cod = soluble_cod + particulate_cod
    total_cod_with_gas = total_liquid_cod + gas_cod

    result = {
        "COD_soluble": soluble_cod,
        "COD_particulate": particulate_cod,
        "COD_liquid_total": total_liquid_cod,
        "COD_gas": gas_cod,
        # Sum of concentrations only: useful for quick debugging, but NOT a
        # physically conservative mass inventory.
        "COD_total_with_gas": total_cod_with_gas,
    }

    if param is not None:
        liquid_inventory = total_liquid_cod * param.V_liq
        gas_inventory = gas_cod * param.V_gas
        total_inventory = liquid_inventory + gas_inventory
        gas_cod_liquid_basis = gas_inventory / param.V_liq

        result.update({
            "COD_liquid_inventory": liquid_inventory,
            "COD_gas_inventory": gas_inventory,
            "COD_total_inventory": total_inventory,
            "COD_gas_liquid_basis": gas_cod_liquid_basis,
            "COD_total_liquid_basis": total_liquid_cod + gas_cod_liquid_basis,
        })

    return result

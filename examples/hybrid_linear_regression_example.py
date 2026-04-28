"""
Example — Tier 1 with a real ML model: replace Rho_2 (carbohydrate
hydrolysis) with a linear-regression model fitted on synthetic data.

Why Rho_2?
----------
Classical ADM1 uses crude first-order hydrolysis kinetics:

    Rho_2 = k_hyd_ch * X_ch     [kgCOD m^-3 d^-1]

This is a well-known limitation — real hydrolysis also depends on
temperature, pH, and the active hydrolytic-biomass population. Rho_2 is
therefore a natural place to slot in a small data-driven correction.

What this file demonstrates
---------------------------
The full hybrid workflow you would follow with any ML backend
(scikit-learn, PyTorch, ONNX, etc.):

    1. Train the model offline. Here, at module import time, on synthetic
       data with np.linalg.lstsq. In a real project this step would load
       saved weights from disk (joblib.load / torch.load / onnxruntime).
    2. Cache the fitted parameters at module level — paid once.
    3. Inside the override function, do *only* inference. The function is
       called once per ODE-solver step (potentially thousands of times per
       simulated day), so keep it lean: no training, no I/O, no large
       allocations.

Wired by:
    configs/Scenario.yaml > scenarios.<your_scenario>.hybrid.rate_overrides:
        Rho_2: "examples.hybrid_linear_regression_example:carbohydrate_hydrolysis_lr"

The model
---------
    Rho_2_hat = b0 + b1 · X_ch + b2 · T_op + b3 · pH

trained against a synthetic "ground truth" that includes Arrhenius-style
temperature and Gaussian pH effects on top of the classical first-order
form. The features are passed in raw — exactly the order listed in the
HybridSpec — so the spec is a faithful description of the model.

Required signature for a rate override
--------------------------------------
def f(state: dict, inhib: dict, param) -> float
    returns a rate in kgCOD m^-3 d^-1.
"""

import numpy as np


# ============================================================================
# 1. Generate synthetic training data and fit the linear regression
# ============================================================================
# In a real project, replace this whole block with:
#     import joblib
#     _MODEL = joblib.load("path/to/rho2_model.pkl")
# and use _MODEL.predict(...) inside the override function.

def _ground_truth_rho2(X_ch, T_op, pH, k_hyd_ch=10.0):
    """
    Synthetic 'true' carbohydrate hydrolysis rate for training-data
    generation. This is what the linear regression will try to mimic.
    """
    T_factor = np.exp(-0.5 * ((T_op - 308.15) / 12.0) ** 2)   # peak at 35 C
    pH_factor = np.exp(-0.5 * ((pH - 7.0) / 0.8) ** 2)        # peak at pH 7
    return k_hyd_ch * X_ch * T_factor * pH_factor


def _fit_linear_regression(n_samples: int = 2000, seed: int = 42):
    """
    Fit Rho_2 ≈ b0 + b1·X_ch + b2·T_op + b3·pH on synthetic data.
    Features are passed raw — same order as the HybridSpec inputs list.
    Returns the (4,) coefficient vector.
    """
    rng = np.random.default_rng(seed)

    # Sample features over realistic operating ranges
    X_ch  = rng.uniform(0.01, 2.0, n_samples)         # kgCOD/m^3
    T_op  = rng.uniform(293.15, 333.15, n_samples)    # 20–60 C
    pH    = rng.uniform(5.5, 8.5, n_samples)          # plant operational range

    # Target with ~2 % multiplicative measurement noise
    y = _ground_truth_rho2(X_ch, T_op, pH)
    y = y * (1.0 + 0.02 * rng.standard_normal(n_samples))

    # Design matrix [1, X_ch, T_op, pH] — raw features, no centring,
    # so saved coefficients work directly with whatever the loader extracts.
    Phi = np.column_stack([
        np.ones(n_samples),
        X_ch,
        T_op,
        pH,
    ])

    coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return coeffs


# Train once at module import time, cache the result.
# Solver hot loop only sees inference.
_COEFFS = _fit_linear_regression()


# ============================================================================
# 2. The override function — called inside the ODE solver
# ============================================================================

def carbohydrate_hydrolysis_lr(state: dict, inhib: dict, param) -> float:
    """
    Linear-regression replacement for Rho_2 (carbohydrate hydrolysis).

    Reads
    -----
    state["X_ch"]    : substrate concentration  [kgCOD/m^3]
    state["S_H_ion"] : pH = −log10(S_H_ion)     [—]
    param.T_op       : operating temperature    [K]

    Returns
    -------
    float : Rho_2 in kgCOD m^-3 d^-1, clipped at zero.
    """
    X_ch = max(state["X_ch"], 0.0)
    T_op = param.T_op
    pH = -np.log10(max(state["S_H_ion"], 1e-14))

    b0, b1, b2, b3 = _COEFFS
    rho2 = b0 + b1 * X_ch + b2 * T_op + b3 * pH
    return max(rho2, 0.0)


if __name__ == "__main__":
    # `python -m examples.hybrid_linear_regression_example` prints the
    # fitted coefficients. Useful to sanity-check what the model learned.
    labels = ["intercept", "X_ch", "T_op [K]", "pH"]
    print("Fitted Rho_2 linear-regression coefficients:")
    for label, coef in zip(labels, _COEFFS):
        print(f"  {label:>10s}  {coef: .6f}")

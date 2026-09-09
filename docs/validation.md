# Validation — the two BSM2 benchmarks

This model is verified against **BSM2** (Benchmark Simulation Model No. 2, Rosen & Jeppsson
2006) with the two tests PyADM1 (Sadrimajd et al. 2021) used. Both are *verifications* — an
independent implementation (MATLAB/Simulink) solved the same equations on the same protocol
and the results must agree — not calibrations against plant data.

| | **A. Steady-state test** (PyADM1 paper Tables 2–4) | **B. Dynamic ring test** (PyADM1 `ringtest.csv`) |
|---|---|---|
| influent | constant BSM2 digester feed | 280 d, 15-min time series, Q(t) 59–466 m³/d |
| flow | **170 m³/d** | Q(t) (MATLAB) / 178.47 const (PyADM1) |
| duration | 200 d (≈ 10 HRT → converged) | 280 d |
| reference | published BSM2 steady state, 14 digits | MATLAB trajectory `Matlabout_dyn.csv` |
| metric | final value, absolute error | 280-d averages + daily RMSE |
| scenario | `BSM2_steady_state` (2 s) | `BSM2_ringtest` (2 min) / `BSM2_ringtest_15min` (50 min) |
| script | `tests/benchmark_bsm2_steady.py` | `tests/benchmark_bsm2.py` |
| result | **max abs err 8e-7** (PyADM1: 1.9e-5); 4e-11 at 400 d | **≤ 0.04 %** on all states (PyADM1: ≤ 2.5 %) |

Run test A first: it is instantaneous, has no time-series alignment issues and a 14-digit target.
Note the flow: 170 m³/d. With 178.4674 (the first value of the dynamic file, hard-coded in
PyADM1) every state is 1–7 % off — that is not a model error.

## A. Steady-state test

```bash
# configs/Scenario.yaml → active_scenario : BSM2_steady_state
python main.py
python tests/benchmark_bsm2_steady.py --results results/dynamic_out.csv
```
Prints the 38 states next to the published BSM2 values and PyADM1's errors; `VERDICT: PASS`
if every |error| ≤ 1e-5. Reference: `tests/data/bsm2_steady_state.csv`; influent block
`constant` in `Influent.yaml`; initial state `BSM2_steady_init`.

## B. Dynamic ring test

### The benchmark = fixed inputs + a reference output

| Item | File | Notes |
|---|---|---|
| Influent, 280 d @ 15 min | `configs/digester_influent.csv` | 26 concentrations + flow `Q` (59–466 m³/d) + T |
| Influent, daily averages | `configs/daily_averages.csv` | derived from the above |
| Initial digester state | `configs/Initial_states.yaml` (`BSM2`) = `tests/data/digester_initial.csv` | S_cation ≈ 0, pH 7.263 |
| Parameters | `configs/adm1_parameters.yaml` | BSM2 values, 35 °C (identical to PyADM1) |
| **Reference output** | `tests/data/Matlabout_dyn.csv` | MATLAB BSM2 ADM1, all 38 states, 15 min |
| PyADM1's own result | `tests/data/ringtest.csv` | yardstick |

### Step by step

1. `configs/Scenario.yaml` → `active_scenario : BSM2_dynamic` (constant q_ad = 178.47,
   PyADM1-identical) **or** `BSM2_ringtest` (dynamic Q(t), closer to MATLAB).
2. `python main.py` → `results/dynamic_out.csv` (280 d, daily).
3. Score:
   ```bash
   python tests/benchmark_bsm2.py --results results/dynamic_out.csv \
          --reference tests/data/Matlabout_dyn.csv --outdir results/benchmark \
          [--dynamic-q]      # add when the scenario uses flow_column: Q
   ```
4. Read `results/benchmark/benchmark_summary.md` and `benchmark_overlay.png`.
   Exit code 0 = PASS (use it in CI).

### Metrics and pass criteria

* **avg err %** — relative error of the 280-day time-integrated average (PyADM1's
  ring-test metric). Tolerance 6 % (pH: 0.05 units).
* **daily nRMSE %** — RMSE of daily means, days 30–280 (start-up excluded), normalised by the
  MATLAB mean. Tolerance 25 % with constant q_ad, 12 % with dynamic Q (daily-averaged feed; S_fa reaches 10 %).
  Never compare instantaneous midnight values: the BSM2 feed has a diurnal cycle.
* **Yardstick** — the `pyadm1_*` columns are what the original PyADM1 obtains; a refactor
  must be at least as close to MATLAB.

### Expected results (this repository, Sept 2026)

| scenario | pH err | S_ac avg err | S_ac daily nRMSE | S_IN avg err | verdict |
|---|---|---|---|---|---|
| `BSM2_dynamic` (const Q) | +0.013 | −2.6 % | 18 % | +0.34 % | PASS (= PyADM1) |
| `BSM2_ringtest` (dyn Q, daily feed) | −0.12 | −3.7 % | 8 % | +0.30 % | PASS |
| `BSM2_ringtest_15min` (dyn Q, 15-min feed) | **+0.0001** | **+0.02 %** | **0.12 %** | **+0.04 %** | **PASS — MATLAB reproduced to ≤ 0.04 % on all 38 states** |
| PyADM1 original | +0.015 | −2.2 % | 17 % | +0.34 % | PASS |

The residual ~2–3 % on VFAs and the 17 % daily RMSE with constant Q come from ignoring the
flow variations, not from the biochemistry; dynamic Q halves the dynamic error. What is
left with dynamic Q is due to `daily_averages.csv` averaging Q and concentrations
separately (Q̄·C̄ ≠ mean(Q·C)); the `BSM2_ringtest_15min` scenario (≈ 50 min; with `dt_out: null` the output has the same 26 881 rows as the influent and the MATLAB file) reproduces MATLAB essentially exactly. Note the reference files are 15-min data with the feed row i held on [t_i, t_{i+1}).

## Regression use (both tests)

Any change to `src/` must keep the verdict PASS. For hybrid models, run the hybrid scenario
with identity hooks: the benchmark table must be identical to `BSM2_dynamic`.

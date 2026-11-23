# 2026-11-06 — Influent row held over each interval (STAMM backlog 63)

## Finding
With a dynamic influent the solver restarts at every influent row t_i. Before this change the
feed was looked up at every solver call with `searchsorted(time, t, side="right") - 1`: over
[t_(i-1), t_i] the row i-1, except the evaluations at t = t_i exactly (the last steps of the
piece), which read row i. The comment of the previous main.py said the opposite (row i over
(t_(i-1), t_i], the convention of the PyADM1 loop, which integrates [t_(n-1), t_n] after
`setInfluent(n)`).

## Change
- stamm-model-core 0.2.2: `OdeModel.before_segment(t_start, t_end)`, called before each piece of
  a continuous run (test in tests/test_forcing.py).
- model-adm1: `influent_hold` in configs/simulation.yaml (and `--influent-hold`); the row is fixed
  for the whole piece in `before_segment`:
  - `start` (default): row i-1, a row holds until the next one;
  - `end`: row i, as the PyADM1 loop;
  - `legacy`: the previous lookup, to reproduce earlier runs (tools/compare_port.py uses it: the
    port alone still changes nothing).
  The value is written in run.json -> calibration.inputs.hold (part of the scenario fingerprint).
- Model version 0.2.0 (configs/model_card.yaml): the dynamic trajectories change slightly
  (constant-influent scenarios do not).

## Study (tools/influent_hold_study.py, report tools/influent_hold_report.md, Camilo's PC)
BSM2 ring test against the MATLAB/Simulink reference, tests/benchmark_bsm2.py --dynamic-q:

| scenario | hold | max abs avg err % | mean abs avg err % | max daily nRMSE % | mean daily nRMSE % | states failing |
|---|---|---|---|---|---|---|
| BSM2_ringtest (daily) | legacy | 3.729 | 1.068 | 10.280 | 2.821 | 0 |
| BSM2_ringtest (daily) | start | 3.728 | 1.068 | 10.280 | 2.821 | 0 |
| BSM2_ringtest (daily) | end | 3.877 | 1.107 | 24.036 | 7.458 | 9 |
| BSM2_ringtest_15min | legacy | 0.014 | 0.003 | 0.150 | 0.048 | 0 |
| BSM2_ringtest_15min | start | 0.012 | 0.003 | 0.151 | 0.049 | 0 |
| BSM2_ringtest_15min | end | 0.012 | 0.003 | 0.149 | 0.049 | 0 |

The MATLAB reference holds each row until the next one: with the daily influent `end` shifts the
feed by one day. At 15 min the three are equivalent. `start` is the best or equal in both tests and
is consistent (no mix of rows within a piece): default (Camilo, 2026-11-06).
results/benchmark_BSM2_ringtest, _15min and _dynamic regenerated with `start`.

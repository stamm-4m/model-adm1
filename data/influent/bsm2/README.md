# BSM2 digester influent

Benchmark Simulation Model No. 2 anaerobic-digester influent (Rosen & Jeppsson, 2006), as used by
PyADM1 and the MATLAB/Simulink BSM2 implementation.

| File | Content | Used by (`configs/influent.yaml`) |
|---|---|---|
| `digester_influent_15min.csv` | 280 d at 15-min resolution (26 881 rows): `time` [d], 26 ADM1 influent concentrations, `Q` [m³/d] (59–466), `T (C)` | `dynamic_15min_Q` |
| `daily_averages.csv` | daily averages of the 15-min file (281 rows) | `dynamic`, `dynamic_Q` |

Units: soluble/particulate COD states in kgCOD.m⁻³, `S_IC` in kmol C.m⁻³, `S_IN` in kmol N.m⁻³,
`S_cation` / `S_anion` in kmol.m⁻³. The extra columns after `T (C)` (acid–base species and gas states)
are not read by the model.

Note: `daily_averages.csv` averages `Q` and the concentrations separately, so the daily load
mean(Q)·mean(C) ≠ mean(Q·C) (see docs/validation_report_2026-09-08.md). For the strictest comparison
with the MATLAB reference use the 15-min file.

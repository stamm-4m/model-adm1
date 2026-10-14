# 5 L lab digester — 2023 campaign

| File | Content | Used by (`configs/influent.yaml`) |
|---|---|---|
| `influent.csv` | 195 days from 2023-03-15, one row per day (row *i* active on [i, i+1) d): `time` [d], 26 ADM1 influent concentrations, `Q` [m³/d], `T` [°C] | `tatiana_lab_5L` |

- Source: `RM_without_Xc/inputs/U_dyn_ref.csv` (Python version of the original lab model).
- The feed is characterised directly in `X_ch` / `X_pr` / `X_li` / `X_I` (`X_xc = 0`):
  use it with `use_xc = 0`.
- `Q` ≈ 1·10⁻⁴ m³/d (0.1 L/d) on fed days, 0 on non-fed days.
- `T` (39 °C) is informative only: the operating temperature is set by the scenario `T_op`.
- Units as in `data/influent/bsm2/README.md`.

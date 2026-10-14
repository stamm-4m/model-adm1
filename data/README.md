# data/

Input data of the model. Configuration (YAML) stays in `configs/`; data files (CSV) live here.

```
data/
└── influent/                  what enters the digester (model INPUT u(t))
    ├── bsm2/                  BSM2 benchmark influent (Rosen & Jeppsson 2006)
    └── lab_5L_2023/           5 L lab digester, 2023 campaign
```

**One folder per data source** (a plant, a digester, a measurement campaign), not per scenario:
several scenarios can share the same influent. The link is:

```
configs/scenario.yaml   influent_mode: <entry>
        └──► configs/influent.yaml   <entry>: file_path: data/influent/<source>/<file>.csv
```

To add a new influent:

1. create `data/influent/<source>/` with the CSV (`time` [d] + the 26 ADM1 influent variables,
   optional flow column `Q` [m³/d]) and a short `README.md` (origin, dates, units, processing);
2. add an entry in `configs/influent.yaml` pointing to it (`file_path`, `time_column`, `flow_column`);
3. select it in a scenario with `influent_mode: <entry>`.

Measured digester data used to *compare* with simulations (biogas, CH₄, pH, …) are not model
inputs: if they are stored in the repository, put them in `data/measurements/<source>/`.
Validation references of the benchmark stay in `tests/data/`.

Every run package (`runs/<run_id>/`) keeps a copy of the influent used, with its `source_path` and
`sha256` in `run.json`, so moving or renaming files here does not break the traceability of past runs.

# ADM1 Reactor Simulation (Adapted from PyADM1)

**Documentation:** see [docs/](docs/) — including [docs/adm1_biology.md](docs/adm1_biology.md), an introduction to the biology of ADM1 written for computer scientists.

---

This project implements a simulation of an anaerobic digestion reactor based on the Anaerobic Digestion Model No.1 (ADM1).

The code is adapted from the original PyADM1 implementation, a Python tool for modeling and simulating anaerobic digestion processes in biogas reactors.

Source repository:
https://github.com/CaptainFerMag/PyADM1

This adaptation restructures and extends the original implementation to support improved configuration handling, data preprocessing, and modular code organization.

## Modifications in This Repository

Compared to the original PyADM1 code, this project introduces:

* Modular project structure
* YAML-based parameter configuration
* CSV-based influent preprocessing
* Improved data management
* Clear separation between model, parameters, and influent data
* Simplified simulation workflow

## 📁 Project Structure
```bash
ADM1/
│
├── main.py                     # Simulation entry point
├── initial_states.py           # Initial state vector loader (38 ADM1 states)
├── requirements.txt            # Python dependencies (pip install -r)
│
├── docs/                       # Extended documentation
│   ├── README.md               # Index of the docs folder
│   └── adm1_biology.md         # ADM1 biology explained for computer scientists
│
├── configs/
│   ├── adm1_parameters.yaml    # Intrinsic ADM1 kinetic / stoichiometric parameters
│   ├── Initial_states.yaml     # Named initial state vectors (BSM2, …)
│   ├── Influent.yaml           # Influent definitions (dynamic CSV or constant)
│   ├── Scenario.yaml           # Active scenario selector + parameter overrides
│   ├── Simulation.yaml         # ODE solver settings, time horizon, output options
│   ├── Calibration.yaml        # Calibration framework (free parameters, bounds)
│   ├── digester_influent.csv   # Raw influent data
│   └── daily_averages.csv      # Daily-averaged influent (BSM2 dynamic)
│
├── src/
│   ├── reactor.py              # ADM1 reactor model (ODE system, mass balances)
│   ├── parameters.py           # Parameter loader with scenario overrides
│   ├── influent.py             # Influent interface (dynamic / constant modes)
│   └── acid_base.py            # Acid-base equilibrium solver (pH, HCO3⁻, NH3)
│
├── plots/
│   ├── plot_biogas.py          # Biogas production diagnostics
│   ├── plot_biomass.py         # Biomass population dynamics
│   └── plot_pH_alkalinity.py   # pH and alkalinity diagnostics
│
└── results/
    └── dynamic_out.csv         # Simulation output
```


## Installation

Python 3.10+ is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/stamm-4m/model-adm1.git
cd model-adm1
```

### 2. Create and activate a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe)**

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 3. Install dependencies

All required packages are pinned in [requirements.txt](requirements.txt):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Environment variables

The simulator is configured entirely through YAML files in [configs/](configs/);
no environment variables are required out of the box. If you want to override
local paths or secrets in the future (e.g. a dataset directory), create a
`.env` file at the project root — it is already listed in `.gitignore`.

### Core dependencies

| Package    | Purpose                              |
| ---------- | ------------------------------------ |
| NumPy      | Numerical operations, state vectors  |
| SciPy      | ODE solver (`solve_ivp`)             |
| Pandas     | Influent CSV / results handling      |
| Matplotlib | Diagnostic plots                     |
| PyYAML     | Configuration loading                |

## Running the Simulation

1️⃣ **Pick a scenario** in `configs/Scenario.yaml` by setting `active_scenario:`
(e.g. `BSM2_dynamic`, `BSM2_constant`, `thermophilic`, `batch_validation`, `pig_slurry_test`).

2️⃣ **Provide influent data** in `configs/Influent.yaml`:
- `dynamic` mode reads a CSV time series (default: `configs/daily_averages.csv`)
- `constant` mode uses fixed values defined directly in the YAML

3️⃣ **Tune solver and output** in `configs/Simulation.yaml` (method, tolerances,
time horizon, output step). Reference parameters live in `configs/adm1_parameters.yaml`
and can be overridden per scenario via `parameter_overrides:` in `Scenario.yaml`.

4️⃣ **Run the simulation**:

```bash
python main.py
```

Results are written to `results/dynamic_out.csv`, and diagnostic figures
(biogas, biomass, pH/alkalinity) are saved to `results/figures/` when
`save_figures: true` in `Simulation.yaml`.

## Simulation Workflow

```bash
Influent Data (CSV)
        │
        ▼
Influent Loader
        │
        ▼
ADM1 Reactor Model
        │
        ▼
ODE Solver (SciPy solve_ivp)
        │
        ▼
Simulation Output (CSV)

```

## 📘 How parameters are structured in YAML files

<details> <summary>📖 Click to expand</summary> <br>

**Authors:**  
- Margaux BONAL — <margaux.bonal@inrae.fr>  
- David Camilo CORRALES — <David-Camilo.Corrales-Munoz@inrae.fr>  

**Date:** April 2026  

---

## Abstract

This document proposes a reorganization of the ADM1 model parameters in order to transform a monolithic configuration file into a modular and structured architecture. The objective is to improve code readability, facilitate the management of simulation scenarios, and enable cleaner parameter calibration.

---

## I. General Overview

This work builds upon the Anaerobic Digestion Model No.1 (ADM1) implementation developed by Sadrmajd et al. [1], with the objective of reorganizing the parameter structure to make it more understandable and intuitive to use.

### a. Context

The current ADM1 implementation relies on a single configuration file:

This file contains in a single location:

- Initial states  
- Stoichiometric parameters  
- Kinetic parameters  
- Physico-chemical parameters  
- Reactor parameters  
- General constants  

This organization allows for quick setup but becomes limiting when:

- Comparing multiple simulation cases  
- Modifying operating conditions  
- Calibrating parameters  
- Enabling or disabling model components  
- Improving readability, robustness, and maintainability  

The objective of this work is therefore to reorganize ADM1 parameter management to provide a clearer, more modular, and easier-to-use configuration system.

---

### Overview of ADM1

ADM1 is a dynamic model describing anaerobic digestion processes. It enables the simulation of:

- Biogas production (methane and carbon dioxide)  
- Digester stability  
- Accumulation of intermediates such as volatile fatty acids (VFA)  

The model also incorporates:

- Inhibition effects (ammonia, pH, hydrogen)  
- Microbial population dynamics  
- Operational conditions (feed rate, temperature)  

---

## New Parameter Organization

Currently, all parameters are stored in a single “all-in-one” YAML file:



### General Organization

The objective is to establish a parameter organization that is both simple and modular. To limit project complexity, the total number of files is intentionally restricted to six.

This organization aims to clearly distinguish:

- The intrinsic structure of the ADM1 model  
- Site- or reactor-specific conditions  
- Operational scenarios  
- Calibration parameters  
- Initial states  
- Dynamic inputs  

---

### Principles of the New Organization

The new organization is based on a clear separation of responsibilities, where each file answers a specific question related to the simulation. This structure improves model understanding and usability while facilitating future extensions.

| Dimension | Question | File |
|----------|--------|------|
| Initial state | What is the starting condition of the digester? | `states/` |
| Inputs | What enters the digester? | `influent/` |
| Simulation scenario | What do we want to test? | `scenarios/` |
| Calibration | Which parameters are adjusted? | `calibration/` |
| Numerical settings | How are the equations solved? | `simulation/` |
| Model parameters | What are the reference ADM1 parameters? | `adm1_parameters.yaml` |

**Table — Correspondence between ADM1 simulation dimensions and configuration files**

This structure provides a simple yet effective separation of roles and forms the foundation of the proposed architecture.

---

### File Structure

The new architecture is based on five specific configuration files and one general parameter file.

**File structure of the new ADM1 parameter organization**

---

## Parameter Files Description

### `configs/adm1_parameters.yaml`

This file serves as the main reference for ADM1 model parameters. It contains all structural parameters defining the model behavior, independent of any specific simulation.

It includes:

- General constants (e.g., R, T_base, p_atm)  
- Stoichiometric parameters  
- Biochemical and kinetic parameters  
- Inhibition parameters  
- Physico-chemical parameters  
- Default reactor parameters  
- Elemental coefficients (carbon, nitrogen)  

It must not contain simulation-dependent elements such as initial states, influent profiles, numerical settings, scenarios, or calibration parameters.

👉 This file acts as a **central and stable parameter library**.

---

### `configs/states/initial_states.yaml`

This file defines the initial conditions of the system, i.e., the state vector at the beginning of the simulation.

It includes:

- Dissolved states  
- Particulate states  
- Ionic states  
- Gas phase components  

It allows multiple initial state configurations:

- Default state  
- BSM2 steady-state benchmark  
- Startup conditions  
- Thermophilic or high-ammonia conditions  

👉 This separation clearly distinguishes model parameters from simulation starting conditions.

---

### `configs/influent/influent.yaml`

This file describes the system inputs, i.e., the composition and characteristics of the feed entering the digester.

It includes:

- Influent definition mode (constant or dynamic)  
- Data file path (for dynamic influent)  
- Expected data structure (columns, variables)  
- Optional constant influent values  

Two main use cases:

- Dynamic influent (time series from CSV, e.g. `daily_averages.csv`)  
- Constant influent (defined directly in YAML)  

👉 This file formalizes influent handling and improves modularity.

---

### `configs/scenarios/scenarios.yaml`

This file defines simulation scenarios and acts as the main entry point for users.

Each scenario includes:

- Initial state  
- Influent mode  
- Simulation duration  
- Parameter overrides  
- Operating conditions  

Examples include:

- Standard mesophilic conditions  
- Thermophilic high-ammonia conditions  
- BSM2 validation  
- Batch or CSTR simulations  
- Scenarios with modified kinetics  

It can also include advanced scenarios integrating thermodynamic constraints, as described in recent studies [2].

👉 This file defines the **simulation context**, not the full parameter set.

---

### `configs/calibration/calibration.yaml`

This file defines the calibration framework, including all elements required for parameter estimation.

It includes:

- List of parameters to calibrate  
- Parameter bounds  
- Observed variables  
- Objective function  
- Optimization method  
- Calibration steps  

👉 This ensures a clear distinction between fixed model parameters and adjustable parameters.

---

### `configs/simulation/simulation.yaml`

This file defines the numerical settings for the simulation.

It includes:

- Solver selection (RK45, BDF)  
- Numerical tolerances (rtol, atol)  
- Simulation duration  
- Output time step  
- Output options  
- ODE/DAE mode  

👉 This separation distinguishes the biophysical model from its numerical resolution.

---

## References

1. Sadrmajd, P., Mannion, P., Howley, E., & Lens, P. N. L.  
   *PyADM1: A Python Implementation of Anaerobic Digestion Model No. 1*  
   https://doi.org/10.1101/2021.03.03.433746  

2. Yeghiazaryan, S., Capson-Tojo, G., Steyer, J.-P., et al.  
   *Modeling Thermophilic Syntrophic VFA Oxidation Using Thermodynamic Principles*  
   https://doi.org/10.1016/j.biortech.2026.134365  



</details
```
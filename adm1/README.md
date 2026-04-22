# ADM1 Reactor Simulation (Adapted from PyADM1)

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
│
├── configs/
│   ├── adm1_parameters.yaml    # Model parameters
│   ├── digester_influent.csv   # Raw influent data
│   └── daily_averages.csv      # Processed influent data
│
├── results/
│   └── dynamic_out.csv         # Simulation output
│
└── src/
    ├── reactor.py              # ADM1 reactor model
    ├── parameters.py           # Parameter loader
    ├── influent.py             # Influent interface
    └── preprocessing_data.py   # Data preprocessing
```


## Requirements

The simulator requires the following packages:

* NumPy
* SciPy
* Pandas
* Matplotlib
* PyYAML

Install them using:

```bash
pip install numpy scipy pandas matplotlib pyyaml
```

### Core dependencies:
* NumPy – numerical operations
* SciPy – ODE solver (solve_ivp)
* Pandas – data processing
* Matplotlib – visualization
* PyYAML – configuration loading

## Running the Simulation
1️⃣ Configure model parameters

```bash
configs/adm1_parameters.yaml
```

## 2️⃣  Provide influent data in:

```bash
configs/
```

## 3️⃣ Run the simulation
```bash
python main.py
```

At the end of the simulation, the results will be written to:

```bash
results/dynamic_out.csv

```

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

## 📘 Project brief – ADM1 Parameter Refactoring

<details> <summary>📖 Click to expand</summary> <br>

**Author:** Margaux BONAL  
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

## Conclusion

This new organization transforms a monolithic structure into a modular and extensible architecture. It improves code readability, facilitates scenario management, and clearly separates model parameters, initial conditions, and numerical settings.

This refactoring provides a strong foundation for future developments, including calibration, optimization, and integration with AI-based tools.

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
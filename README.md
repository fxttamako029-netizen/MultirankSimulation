# MultiRank Simulation in International Financial Networks

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Project](https://img.shields.io/badge/Project-Honours%20Thesis-6f42c1)
![Data](https://img.shields.io/badge/Data-BIS%20Cross--border%20Banking-0A66C2)
![Status](https://img.shields.io/badge/Status-Research%20Repository-success)

Code, data, figures, and supporting materials for my Honours thesis on **systemic risk and financial contagion in multiplex international banking networks**.

The project builds directed financial networks from BIS cross-border banking exposures, applies **MultiRank** to estimate the systemic importance of countries and network layers, and runs stylised stress-testing experiments involving default contagion, heterogeneous buffers, credit crunches, and targeted bailouts.

> **Research use:** These simulations are stylised network stress tests. They are not forecasts of real-world sovereign or banking-sector defaults.

## What this repository does

The workflow moves from descriptive network analysis to increasingly rich systemic-risk experiments:

1. Construct directed weighted country-level banking networks from BIS data.
2. Build a two-layer multiplex network using total claims and loans/deposits.
3. Apply MultiRank to jointly estimate country centrality and layer importance.
4. Introduce horizontal and vertical default-contagion thresholds.
5. Add external-asset assumptions and heterogeneous buffers.
6. Add a credit-crunch channel through funding withdrawal.
7. Compare no-intervention and targeted-bailout scenarios.

The empirical focus is the period surrounding the **2007–2009 Global Financial Crisis**.

## Research questions

This repository was developed to explore questions such as:

- Which countries are systemically important when several financial layers are considered simultaneously?
- How do country rankings change across pre-crisis, crisis, and post-crisis periods?
- How does multiplex network structure affect the propagation of financial distress?
- How sensitive is contagion to different capital/buffer assumptions?
- How much additional systemic stress can a credit crunch create?
- Can targeted interventions reduce the number of defaults in a stylised contagion model?

## Data

The `data/` directory contains BIS cross-border banking exports used by the simulations:

```text
2006Q1-2007Q2.csv
2007Q1-2007Q4.csv
2007Q3-2008Q4.csv
2008Q1-2008Q4.csv
2009Q1-2009Q4.csv
```

The network direction is defined as:

```text
Reporting country  ---- exposure ---->  Counterparty country
```

For a given layer and quarter, the adjacency matrix is interpreted as

```text
A[i, j] = exposure from reporting country i to counterparty country j
```

Self-loops are excluded. Missing, suppressed, or non-positive observations are treated as zero exposure when the matrices are constructed.

## Multiplex network

The main MultiRank and contagion experiments use two analytical layers:

| Layer | BIS instrument |
| --- | --- |
| Layer 0 | All instruments |
| Layer 1 | Loans and deposits |

The layers are stored as a list of adjacency matrices:

```python
A_layers = [A_layer0, A_layer1]
```

`All instruments` and `Loans and deposits` are analytical layers and should not be interpreted as mutually exclusive asset categories.

## MultiRank

MultiRank jointly estimates two quantities:

- `x`: the relative systemic importance of each country;
- `z`: the relative importance of each financial layer.

The country and layer scores are updated iteratively. Countries that are important in influential layers receive higher scores, while layers connected to highly ranked countries gain more influence.

The implementation uses a PageRank-style damping parameter with the default

```python
kappa = 0.85
```

and iterates until the requested number of iterations or convergence criterion is reached, depending on the script.

## Contagion model

Later experiments use a stylised double-threshold multiplex contagion mechanism.

### Horizontal contagion

A country becomes distressed in a particular layer when losses in that layer exceed a layer-specific buffer:

```text
Layer loss / Layer assets > b
```

### Vertical contagion

A country becomes distressed across the multiplex network when combined losses exceed its total buffer:

```text
Total loss / Estimated total assets > c
```

Later scripts extend this baseline with external assets, heterogeneous country buffers, a liquidity/credit-crunch shock, and targeted bailout protection.

## Repository structure

```text
MultirankSimulation/
├── data/                         # BIS data used by the simulations
├── Simulation/                   # Analysis and stress-testing scripts
│   ├── 01_build_network.py
│   ├── 02_descriptive_plots.py
│   ├── 03_multirank_analysis.py
│   ├── 04_contagion_baseline.py
│   ├── 05_contagion_external_assets.py
│   ├── 06_bailout_experiment.py
│   ├── 07_heterogeneous_buffers.py
│   ├── 08_credit_crunch.py
│   ├── 09_bailout_credit_crunch_experiments.py
│   └── 10_multiplex_3d_visualization.py
├── Paper/                        # Background/reference literature
├── Honours_thesis Xiangtai Feng.pdf
├── Presentation.pdf
├── CITATION.cff
├── requirements.txt
└── README.md
```

## Script guide

| Script | Purpose |
| --- | --- |
| `01_build_network.py` | Build and visualise representative directed weighted BIS networks. |
| `02_descriptive_plots.py` | Produce descriptive exposure plots and representative snapshots. |
| `03_multirank_analysis.py` | Construct the two-layer multiplex network and compute MultiRank country/layer scores. |
| `04_contagion_baseline.py` | Baseline double-threshold contagion experiment. |
| `05_contagion_external_assets.py` | Contagion model with an external-asset assumption and pre/post-shock MultiRank comparison. |
| `06_bailout_experiment.py` | Compare contagion with and without a targeted bailout. |
| `07_heterogeneous_buffers.py` | Assign different buffers to core and other countries and simulate contagion. |
| `08_credit_crunch.py` | Compare solvency contagion with solvency + liquidity/credit-crunch contagion. |
| `09_bailout_credit_crunch_experiments.py` | Run the most comprehensive experiment matrix with shocks, credit crunch, heterogeneous buffers, and bailouts. |
| `10_multiplex_3d_visualization.py` | Visualise the empirical multiplex network in 3D and display MultiRank results. |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/fxttamako029-netizen/MultirankSimulation.git
cd MultirankSimulation
```

### 2. Create a virtual environment

Python **3.10+** is recommended.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run an analysis

Start with the MultiRank analysis:

```bash
python Simulation/03_multirank_analysis.py
```

Run the baseline contagion model:

```bash
python Simulation/04_contagion_baseline.py
```

Run heterogeneous-buffer contagion:

```bash
python Simulation/07_heterogeneous_buffers.py
```

Run the credit-crunch comparison:

```bash
python Simulation/08_credit_crunch.py
```

Run the full bailout + credit-crunch experiment matrix:

```bash
python Simulation/09_bailout_credit_crunch_experiments.py
```

Most scripts print results to the terminal and display figures with Matplotlib.

## Data paths

The scripts use a repository-relative data path:

```python
DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
```

Therefore, the repository can be cloned to any location without editing a machine-specific absolute path.

## Main experiment parameters

The main simulation parameters are intentionally kept near the top of each script so that scenarios can be modified easily.

Typical examples include:

```python
TARGET_QUARTER = pd.Timestamp("2008-12-31")
VICTIM_COUNTRY = "US"
CRUNCH_RATIO = 0.5
```

Homogeneous-buffer experiments use parameters such as:

```python
BUFFER_LAYER = 0.05
BUFFER_TOTAL = 0.10
```

Heterogeneous-buffer experiments distinguish a group of core financial countries from the remaining countries, for example:

```python
BUFFER_LAYER_CORE = 0.03
BUFFER_TOTAL_CORE = 0.05
BUFFER_LAYER_REST = 0.08
BUFFER_TOTAL_REST = 0.12
```

Bailout experiments allow the initial shock and intervention target(s) to be changed directly in the experiment configuration.

## Suggested workflow for reproducing the thesis logic

A useful order is:

```text
01_build_network.py
        ↓
02_descriptive_plots.py
        ↓
03_multirank_analysis.py
        ↓
04_contagion_baseline.py
        ↓
05_contagion_external_assets.py
        ↓
06_bailout_experiment.py
        ↓
07_heterogeneous_buffers.py
        ↓
08_credit_crunch.py
        ↓
09_bailout_credit_crunch_experiments.py
```

The files are separate research scripts rather than a single Python package. This preserves the step-by-step development of the thesis experiments and makes individual assumptions easy to inspect.

## Interpretation and limitations

The model should be read as a **stylised systemic-risk experiment**, not a calibrated forecasting model.

Important assumptions include:

- country-level banking systems are represented as nodes rather than individual banks;
- BIS cross-border exposures are used as network links;
- missing/suppressed observations are treated as zero when constructing the matrices;
- self-exposures are excluded;
- capital/buffer thresholds are modelling assumptions rather than direct regulatory capital estimates;
- later simulations estimate total assets using an external-asset assumption;
- the credit-crunch channel is represented as a simplified funding-withdrawal shock;
- a bailout is represented by preventing selected nodes from entering or remaining in the distressed state;
- results are intended for comparative network analysis rather than real-world prediction.

## Thesis and presentation

- [Honours thesis (PDF)](./Honours_thesis%20Xiangtai%20Feng.pdf)
- [Thesis presentation (PDF)](./Presentation.pdf)

For the mathematical motivation, literature review, modelling assumptions, and interpretation of the experiments, please refer to the thesis itself.

## Citation

If you use this repository or build on the simulations in academic work, please cite the thesis/repository. A machine-readable `CITATION.cff` file is included so GitHub can display a **Cite this repository** option.

## Author

**Xiangtai Feng**

Honours research project on MultiRank, multiplex financial networks, systemic risk, and financial contagion.

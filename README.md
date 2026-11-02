# SEAMLESS

This repository provides the code and data for **SEAMLESS**, a locally adaptive network dismantling method based on edge-neighborhood dissimilarity.

This work is from **CoMuNe Lab**, led by [Prof. Manlio De Domenico](https://manliodedomenico.com/).
## Code

- **`seamless_robustness.py`** — Implements SEAMLESS and five comparison strategies (random, degree, adaptive degree, betweenness, and adaptive betweenness). Evaluates network fragmentation through the largest and second-largest connected components (`S1/N` and `S2/N`).
- **`run_seamless.sh`** — Runs network dismantling experiments on a selected network.
- **`batch.sh`** — Runs experiments for multiple network instances in parallel.
- **`plot_seamless.py`** — Plots component-size curves from the computed `summary.csv` results.

## Figure

The `Figure/` folder contains scripts used to produce the figures and analyze the corresponding experiments:

- **`Figure_seamless.py`** — Illustrates the SEAMLESS procedure and example dismantling results.
- **`Figure_synthetic20.py`** — Analyzes SEAMLESS on synthetic networks, including local exploration and network fragmentation.
- **`Figure_synthetic20_data.py`** — Generates the synthetic network dismantling data used in the analysis.
- **`Figure_Metabolic.py`** — Analyzes dismantling results and biological characteristics of metabolic networks.
- **`Figure_PPI.py`** — Analyzes dismantling results and biological characteristics of protein–protein interaction (PPI) networks.
- **`Figure_bio_curve.py`** — Plots dismantling curves for metabolic and PPI networks.
- **`Figure_evolution.py`** — Visualizes network evolution, fragility, and efficiency.
- **`Figure_evolution_data.py`** — Generates network evolution simulation data.
- **`Figure_SI1.py`** — Compares synthetic-network dismantling performance, runtime, and computational complexity.
- **`Figure_SI1_data.py`** — Measures synthetic-network runtime for the six dismantling strategies.

## Data

The `Data/` folder contains network datasets and associated biological information:

- **Synthetic networks:** BA, ER, HSBM4, LFR, RGG, SBM4, SBM8, SBM16, SF2, and WS (`*_N256/` folders).
- **Metabolic networks:** `Metabolic.zip`, with accompanying organism and network information in `Metabolic_SI/`.
- **PPI networks:** `PPI_part1.zip` and `PPI_part2.zip`, with accompanying species information in `PPI_SI/`.

Network files use edge-list (`.edges`) format. The main program reads undirected, unweighted networks.

**Note:** Some figure scripts expect precomputed results and locally configured paths. Adjust their input/output directories before running.


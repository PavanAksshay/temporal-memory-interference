# Reproducibility Checklist and Specifications

This document specifies the complete computational environment, hyperparameters, data pipelines, seed schedules, and execution commands required to reproduce all figures, tables, and statistics in the paper.

---

## 1. Computational Environment & Software Versions

- **Operating System**: macOS 14+ / Linux Ubuntu 22.04 LTS
- **Python Version**: Python 3.10.x / 3.11.x
- **Core Dependencies**:
  - `torch == 2.2.0` (or `2.1.x`)
  - `numpy == 1.26.4`
  - `scipy == 1.12.0`
  - `pandas == 2.2.0`
  - `scikit-learn == 1.4.0`
  - `matplotlib == 3.8.2`
  - `seaborn == 0.13.2`
  - `pytest == 8.0.0`
- **Hardware Footprint**: Standard workstation CPU / single consumer GPU (NVIDIA RTX 3090 / Apple M-series). Total execution time for all experiments across 10 seeds: $< 45$ minutes.

---

## 2. Benchmark Generator Parameters (Dynamic SBM)

- **Node Set**: $N = 300$ static nodes.
- **Communities**: $K = 3$ clusters ($100$ nodes per cluster).
- **Global Density**: $\rho = 0.10$.
- **Affiliation Contrast**: $M = 4.0$ ($p_{in} \approx 0.203, p_{out} \approx 0.051$).
- **Regime Transition Schedule**:
  - $T_A = 100$ steps ($\lambda_A = 0.35$).
  - $T_B \in \{25, 50, 100, 200\}$ steps ($\lambda_B = 0.15$).
  - $T_{eval} = 50$ steps ($\lambda_A = 0.35$).
- **Random Seed Range**: 10 independent evaluation seeds: `seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`.

---

## 3. Model Architecture and Hyperparameters

### Continuous TGN
- **Memory Dimension ($d_m$)**: $128$ (default); grid: $\{16, 32, 64, 128, 256\}$.
- **Node Embedding Dimension**: $128$.
- **Time Embedding Dimension**: $32$ (harmonic Fourier encoding).
- **Recurrent Cell**: Standard GRU cell with message aggregator: $\text{MLP}([M_u \,\|\, M_v \,\|\, \Delta t \,\|\, e])$.
- **GNN Layers**: 1-layer temporal graph attention / sum aggregator ($k=1$ hop neighborhood).
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$).
- **Batch Size**: 200 events.
- **Early Stopping**: Patience $= 15$ epochs on validation AP.

### Baseline Implementations
- **EdgeBank**:
  - `EdgeBank_AllHistory`: Unbounded hash lookup $\mathcal{E}_{hist}$.
  - `EdgeBank_Bounded`: Hash lookup restricted to $\tau \in [1, T_A]$.
- **Historical Retrieval**: Non-parametric cosine similarity over normalized snapshot degree/affinity profiles with top-1 retrieval.
- **Random Retrieval**: Top-1 retrieval from uniformly sampled historical snapshot index $\tau \in [1, T_A]$.
- **Historical Oracle**: Evaluation of test candidate pairs directly under ground-truth affinity matrix $S_A$.

---

## 4. Evaluation Protocol and Integrity Checks

- **Positive / Negative Ratio**: Exactly 1:1 balanced candidate set per evaluation step.
- **Negative Sampling**: Uniform random non-edge sampling from $\mathcal{V} \times \mathcal{V} \setminus \mathcal{E}_t$.
- **Leakage Audits**:
  - Temporal leakage: Zero future events visible ($\tau < t$).
  - Target leakage: Evaluated positive edges omitted from message passing at step $t$.
  - Candidate parity: Identical candidate sets evaluated across all 7 methods.

---

## 5. Artifact and Figure Reproduction Commands

All figures and summary tables can be reproduced directly from the command line:

```bash
# Run unit tests verifying pipeline integrity and zero leakage
pytest tests/test_phase4_5.py

# Execute full 10-seed falsification and baseline comparison sweep
python experiments/phase4_5_falsification.py

# Generate final publication-quality figures
python -c "
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('results/phase4_5/processed/final_summary.csv')
print(df[df['dataset'] == 'Synthetic_Hard_Benchmark'])
"
```

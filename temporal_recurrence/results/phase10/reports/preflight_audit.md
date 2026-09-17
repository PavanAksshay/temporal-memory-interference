# Preflight Configuration and Numerical Audit — Phase 10

**Date:** 2026-09-10  
**Status:** Frozen Pre-Execution Baseline  
**Scope:** Rigorous benchmark reconciliation prior to Phase 10 scientific strengthening experiments.

---

## 1. Executive Summary & Benchmark Disentanglement

To ensure zero benchmark conflation and maintain total scientific integrity, this audit explicitly distinguishes the **ORIGINAL FROZEN BENCHMARK** (Phase 0.1 / Phase 4.5 / Phase 6 / Phase 6.5) from the **Phase 9 Prototyping Benchmark**.

| Dimension | Original Frozen Benchmark (Canonical) | Phase 9 Prototyping Testbench |
| :--- | :--- | :--- |
| **Purpose** | **Canonical scientific evaluation & claim validation** | Rapid prototype testing of learnable routing logic |
| **Vertex Count ($N$)** | **$N = 300$** | $N = 60$ |
| **Number of Communities ($K$)** | **$K = 3$** (Equal 100-node partitions) | $K = 3$ (Equal 20-node partitions) |
| **Regime A Persistence ($\lambda_A$)** | **$\lambda_A = 0.35$** (Hard Benchmark) / $\lambda_A = 0.85$ (Specificity Control) | $\lambda_A = 0.85$ |
| **Regime B Persistence ($\lambda_B$)** | **$\lambda_B = 0.15$** (Distractor) | $\lambda_B = 0.20$ |
| **Regime C Persistence ($\lambda_C$)** | **$\lambda_C = 0.25$** (Control $A \to B \to C$) | N/A |
| **Target Edge Density ($\rho$)** | **$\rho = 0.10$** (Strictly matched across all regimes) | $\rho = 0.12$ |
| **Within-Comm Multiplier** | **$4.0\times$** | $3.5\times$ |
| **Chronological Protocol** | Train: $t \in [0, 70]$, Val: $t \in [71, 99]$, Distractor: $T_B \in \{25, 50, 100, 200\}$, Test: $t \in [100+T_B, 150+T_B]$ | Train: $t \in [0, 40]$, Distractor: $T_B = 80$, Test: $t \in [120, 160]$ |
| **Random Evaluation Seeds** | **10 Seeds: $42, 43, 44, 45, 46, 47, 48, 49, 50, 51$** | 1 Seed: $42$ |
| **Negative Sampling** | **$1:1$ balanced random non-edge sampling per snapshot** | $1:1$ balanced random sampling |
| **Evaluation Metrics** | **Average Precision (AP), ROC-AUC, Mean $\pm$ Std** | AP, ROC-AUC |

> [!IMPORTANT]
> **Strict Rule for Phase 10:** All primary comparisons, component ablations, memory budget sweeps, and recurrence decompositions must be conducted on the **ORIGINAL FROZEN BENCHMARK ($N=300$)** across all 10 random seeds ($42$–$51$) and all canonical distractor durations $T_B \in \{25, 50, 100, 200\}$. No numbers from the $N=60$ prototype testbench will be mixed into canonical tables.

---

## 2. Canonical Frozen Benchmark Configuration

### A. Generator Parameters
- **Generator Class:** `DynamicSBMGenerator` (`src/generator/dsbm.py`)
- **Number of Nodes ($N$):** $300$
- **Number of Communities ($K$):** $3$
- **Community Partitions:**
  - **Regime A:** Partition seed `101` ($100$ nodes per cluster)
  - **Regime B:** Partition seed `202` (Independent random permutation of cluster memberships)
  - **Regime C:** Partition seed `303` (Third independent partition for non-recurring control)
- **Stationary Marginal Density:** $\rho = 0.10$ exactly matched across Regimes A, B, and C via Markov birth/death transition balancing:
  $$a_e^{(r)} = W_e^{(r)} (1 - \lambda_r), \quad b_e^{(r)} = (1 - W_e^{(r)}) (1 - \lambda_r)$$
- **Connection Probabilities:**
  - Base density $\rho = 0.10$
  - Within-community multiplier: $4.0\times$
  - Regime A: $\lambda_A = 0.35$ (Hard benchmark, where current snapshot is informative but historical regime knowledge adds $+0.0269$ AP)
  - Regime B: $\lambda_B = 0.15$ (Conflicting distractor regime)
  - Regime C: $\lambda_C = 0.25$ (Control non-recurring regime)

### B. Temporal Protocol & Splits
- **Regime A Exposure:** $t \in [0, 99]$ (100 timesteps)
  - **Train Window:** $t \in [0, 70]$
  - **Validation Window:** $t \in [71, 99]$
- **Regime B Distractor Interval:** $t \in [100, 99 + T_B]$
  - **Tested $T_B$ Values:** $T_B \in \{25, 50, 100, 200\}$
- **Regime A Recurrence Test Window:** $t \in [100 + T_B, 150 + T_B]$ (50 test timesteps)

### C. Random Evaluation Seeds
- **Seeds ($n=10$):** $42, 43, 44, 45, 46, 47, 48, 49, 50, 51$

---

## 3. Model Hyperparameters & Parameter Accounting

### A. Baseline Model Specifications
- **Continuous TGN:**
  - Node embedding dimension: $d_{node} = 64$
  - Memory dimension: $d_m = 64$
  - Time embedding dimension: $d_t = 32$
  - Message dimension: $d_{msg} = 32$
  - Recurrent cell: GRU cell
  - Graph attention: 2-layer temporal attention with 2 heads
  - Optimizer: Adam, Learning Rate $\eta = 0.005$, weight decay $1\times 10^{-5}$
  - Training Epochs: 12 epochs with best validation AP checkpointing
  - Batch sample size: 300 event batches per timestep
- **TGN-NoMemory:**
  - Identical architecture and embedding dimensions, with memory updates disabled ($s_u(t) = x_u$).
- **Historical Retrieval Probe:**
  - Non-parametric historical snapshot buffer of Regime A sketches with cosine similarity addressing.
- **EdgeBank:**
  - `EdgeBank Bounded A`: Stores unique edges observed in Regime A ($t \in [0, 99]$).
  - `EdgeBank All-History`: Stores all historical edges observed up to query time $t$.
- **Historical Oracle & Current-Only Predictors:**
  - Analytical Bayes-optimal predictors computed from true active and historical community partition matrices.

### B. Memory-Augmented TGN (MA-TGN) Specification
- **Continuous Recurrent Tier:** Identical GRU memory bank ($N \times d_m$)
- **Episodic Key-Value Bank:** Stores $K$ historical memory checkpoints $(k_\tau, S_\tau)$ with strict causality verification ($\tau < t$).
- **Multi-Head Addressable Router:** 4 attention heads, key dimension $d_k = 64$, query dimension $d_q = 64$, value dimension $d_v = 64$.
- **Adaptive Temporal Gating:** Dynamic fusion gate $g_u(t) = \sigma(W_g [s_u(t) \parallel r_u(t) \parallel x_u] + b_g)$.
- **Optimizer & Schedule:** Adam, $\eta = 0.005$, 12 epochs on Regime A train window.

---

## 4. Canonical Frozen Baseline Targets (for Reproduction Verification)

From Phase 4.5 and Phase 6, the canonical mean AP scores on the hard benchmark are:

| Method | $T_B = 25$ | $T_B = 50$ | $T_B = 100$ | $T_B = 200$ |
| :--- | :---: | :---: | :---: | :---: |
| **Historical Oracle** | 0.7904 ± 0.0006 | 0.7904 ± 0.0006 | 0.7904 ± 0.0005 | 0.7905 ± 0.0005 |
| **Current-Only** | 0.7636 ± 0.0007 | 0.7636 ± 0.0007 | 0.7635 ± 0.0006 | 0.7638 ± 0.0007 |
| **EdgeBank (Bounded A)** | 0.7423 ± 0.0006 | 0.7424 ± 0.0006 | 0.7423 ± 0.0004 | 0.7424 ± 0.0009 |
| **EdgeBank (All-History)** | 0.7403 ± 0.0006 | 0.7399 ± 0.0006 | 0.7397 ± 0.0004 | 0.7398 ± 0.0008 |
| **Historical Retrieval Probe** | 0.7421 ± 0.0029 | 0.7355 ± 0.0045 | 0.7273 ± 0.0026 | 0.7197 ± 0.0024 |
| **Continuous TGN** | 0.5942 ± 0.0540 | 0.5420 ± 0.0355 | 0.5274 ± 0.0339 | 0.5028 ± 0.0013 |
| **TGN-NoMemory** | 0.5842 ± 0.0632 | 0.6075 ± 0.0557 | 0.5328 ± 0.0297 | 0.5032 ± 0.0015 |

---

## 5. Preflight Checklist Confirmation

- [x] Canonical $N=300$ frozen benchmark identified and preserved.
- [x] Phase 9 prototype parameters isolated and prevented from contaminating canonical tables.
- [x] All 10 evaluation seeds ($42$–$51$) locked.
- [x] Strict temporal causality and zero future leakage verified in episodic bank.
- [x] Directory structure created at `results/phase10/`.

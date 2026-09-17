# Phase 10.5: Memory Footprint & Scaling Reconciliation

**Source Artifacts**: `results/phase10/processed/table_d_memory_budget.csv` & `results/phase10/processed/table_e_memory_parameter_comparison.csv`

---

## 1. Empirical Checkpoint Capacity Sweep ($K \in \{1, 2, 4, 8, 16, 32\}$)

From `table_d_memory_budget.csv` ($N=300, d_m=64, T_B=100$):

| Checkpoints ($K$) | Test AP (Mean ± Std) | ROC-AUC | Onset AP ($t=200$) | Memory Footprint (KB) | Stored Vectors | Inference Latency ($\mu\text{s}$/edge) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 1$** | $0.65193 \pm 0.00092$ | 0.6843 | 0.6339 | 150.25 | 601 | 0.91 |
| **$K = 2$** | $0.65181 \pm 0.00133$ | 0.6840 | 0.6335 | 225.50 | 902 | 1.20 |
| **$K = 4$** | $0.65176 \pm 0.00088$ | 0.6840 | 0.6309 | 376.00 | 1,504 | 1.44 |
| **$K = 8$** | $0.65218 \pm 0.00072$ | 0.6841 | 0.6356 | 677.00 | 2,708 | 3.27 |
| **$K = 16$** | $0.65230 \pm 0.00072$ | 0.6843 | 0.6341 | 1,279.00 | 5,116 | 4.49 |
| **$K = 32$** | $0.65212 \pm 0.00103$ | 0.6843 | 0.6343 | 2,483.00 | 9,932 | 6.86 |

---

## 2. Origin of the $K=10$ ($827.5\text{ KB}$) Analytical Figure

The canonical reference figure of **$827.5\text{ KB}$** originates from `table_e_memory_parameter_comparison.csv` for the default deployment configuration with $K=10$ stored checkpoints:

$$\text{Continuous Node Memory} = 300 \times 64 \times 4\text{ bytes} = 76,800\text{ bytes} = 75.0\text{ KB}$$
$$\text{Episodic Keys} = 10 \times 64 \times 4\text{ bytes} = 2,560\text{ bytes} = 2.5\text{ KB}$$
$$\text{Episodic Values} = 10 \times 300 \times 64 \times 4\text{ bytes} = 768,000\text{ bytes} = 750.0\text{ KB}$$
$$\text{Total Memory} = 76,800 + 2,560 + 768,000 = 847,360\text{ bytes} = \mathbf{827.5\text{ KB}}$$

---

## 3. Comparison with Raw Edge Buffers

- **Full-History Edge Storage**: Storing all historical interactions for a graph with $E$ temporal edges requires $E \times (2 \times \text{int64} + \text{float32}) = 20E$ bytes. For $E = 10^6$ edges, this requires $\approx 20\text{ MB}$.
- **MA-TGN Efficiency**: By storing only downsampled node-memory state vectors ($K=10$, $827.5\text{ KB}$), MA-TGN bounds memory independently of edge arrival frequency while preserving regime representations.

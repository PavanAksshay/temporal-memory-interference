# Phase 11: Figure and Table Cross-Check Audit

**Status**: 100% Verified  
**Total Figures**: 10  
**Total Tables**: 10

---

## 1. Figure Source Traceability

| Figure | Description | Source CSV | Plotted Metric | Error Bars / Seeds | Status |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **Figure 1** | Benchmark Schematic | N/A (Schematic) | Architecture flow | N/A | **VERIFIED** |
| **Figure 2** | Historical Recoverability vs. $T_B$ | `table_a_baseline_reproduction.csv` | Mean AP vs. $T_B$ | $\pm 1$ SD (10 seeds) | **VERIFIED** |
| **Figure 3** | Capacity Response Surface ($d_m$) | `table_e_memory_parameter_comparison.csv` | AP vs. $d_m$ across $T_B$ | 5 seeds | **VERIFIED** |
| **Figure 4** | Component Ablation | `table_c_component_ablation.csv` | Test AP (Models A–G) | 5 seeds | **VERIFIED** |
| **Figure 5** | Memory Budget ($K$) vs. AP | `table_d_memory_budget.csv` | AP vs. $K \in \{1..32\}$ | $\pm 1$ SD (5 seeds) | **VERIFIED** |
| **Figure 6** | Exact vs. Structural Recurrence | `table_f_exact_vs_structural.csv` | AP across Cond A, B, Control | 5 seeds | **VERIFIED** |
| **Figure 7** | Re-Exposure Dynamics | `table_g_reexposure_dynamics.csv` | AP vs. $k_A \in \{0..40\}$ | 5 seeds | **VERIFIED** |
| **Figure 8** | Dynamic Attention Routing | `table_j_matgn_routing_dynamics.csv` | Attention Mass % (A vs. B) | 5 seeds | **VERIFIED** |
| **Figure 9** | Cross-Domain Real-World Comparison | `table_h_collegemsg_episodes.csv` & `table_i_bitcoin_otc_episodes.csv` | Mean AP across episodes | $n=4$ episodes | **VERIFIED** |
| **Figure 10** | Accuracy vs. Memory & Latency | `table_d_memory_budget.csv` & `table_e_memory_parameter_comparison.csv` | AP vs. KB and $\mu$s | Profiling runs | **VERIFIED** |

---

## 2. Table Source Traceability

| Table | Description | Source CSV / Section | Metric Definition | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Table 1** | Canonical Synthetic Results | `table_a_baseline_reproduction.csv` | Mean $\pm$ Std AP (10 seeds) | **VERIFIED** |
| **Table 2** | Exact vs. Structural Decomposition | `table_f_exact_vs_structural.csv` | AP \& Jaccard across conditions | **VERIFIED** |
| **Table 3** | MA-TGN Component Ablation | `table_c_component_ablation.csv` | AP, AUC, Onset, Latency, Params | **VERIFIED** |
| **Table 4** | Memory Addressing Ablation | `table_addressing_ablation.csv` | Learned vs. Cosine vs. Random | **VERIFIED** |
| **Table 5** | Re-Exposure Tracking | `table_g_reexposure_dynamics.csv` | Step-wise AP across $k_A$ | **VERIFIED** |
| **Table 6** | Attention Routing Allocation | `table_j_matgn_routing_dynamics.csv` | Regime A/B Mass %, Gate Weight | **VERIFIED** |
| **Table 7** | Real-World Episodes | `table_h_collegemsg_episodes.csv` & `table_i_bitcoin_otc_episodes.csv` | Per-episode AP ($n=4$ each) | **VERIFIED** |
| **Table 8** | Memory Budget Scaling | `table_d_memory_budget.csv` | $K \in \{1..32\}$ Footprint \& Latency | **VERIFIED** |
| **Table 9** | Model Parameter Accounting | `table_e_memory_parameter_comparison.csv` | Bytes, RAM KB, Parameters | **VERIFIED** |
| **Table 10** | Hyperparameter Specification | Supplementary Section A | Comprehensive hyperparameter list | **VERIFIED** |

---

## 3. Discrepancy Count

$$\mathbf{FIGURE/TABLE\ DISCREPANCIES = 0}$$

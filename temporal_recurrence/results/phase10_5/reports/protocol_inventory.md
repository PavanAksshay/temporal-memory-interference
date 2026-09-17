# Phase 10.5: Protocol Inventory

**Protocol Count**: Exactly 14 Documented Empirical Protocols.  
**Execution Source**: `experiments/phase10_scientific_strengthening.py`

---

## 1. Complete Empirical Protocol Matrix

| Protocol # | Protocol Name | Scientific Purpose | Statistical Unit | Primary Artifact | Status |
| :---: | :--- | :--- | :---: | :--- | :---: |
| **1** | Canonical Baseline Reproduction | Verify deterministic baseline and statistical probe stability | 10 Seeds (42–51) | `table_a_baseline_reproduction.csv` | **VERIFIED** |
| **2** | MA-TGN Frozen Evaluation | Benchmark MA-TGN across distractor durations ($T_B \in \{25, 50, 100, 200\}$) | 10 Seeds (42–51) | `table_b_canonical_sweep.csv` | **VERIFIED** |
| **3** | Component Ablation Suite | Isolate recurrent memory, episodic buffer, learned attention, and gating (Models A–G) | 5 Seeds (42–46) | `table_c_component_ablation.csv` | **VERIFIED** |
| **4** | Memory Budget Capacity Sweep | Characterize performance scaling with checkpoint capacity ($K \in \{1, 2, 4, 8, 16, 32\}$) | 5 Seeds (42–46) | `table_d_memory_budget.csv` | **VERIFIED** |
| **5** | Recurrence Decomposition | Distinguish exact edge repetition from structural community recurrence | 5 Seeds (42–46) | `table_f_exact_vs_structural.csv` | **VERIFIED** |
| **6** | Re-Exposure Dynamics | Characterize step-wise recovery lag across renewed interactions ($k_A \in \{0, 1, 5, 10, 25, 40\}$) | 5 Seeds (42–46) | `table_g_reexposure_dynamics.csv` | **VERIFIED** |
| **7** | Attention Routing Analysis | Measure episodic attention mass allocation during Regime A, B, Onset, and Steady State | 5 Seeds (42–46) | `table_j_matgn_routing_dynamics.csv` | **VERIFIED** |
| **8** | Memory Addressing Ablation | Compare learned multi-head attention against cosine similarity, random, and most-recent | 5 Seeds (42–46) | `table_addressing_ablation.csv` | **VERIFIED** |
| **9** | SNAP CollegeMsg Benchmark | Evaluate multi-week recurrence across 4 human communication episodes | 4 Episodes | `table_h_collegemsg_episodes.csv` | **VERIFIED** |
| **10** | SNAP Bitcoin-OTC Benchmark | Evaluate cross-domain recurrence across 4 financial trust episodes | 4 Episodes | `table_i_bitcoin_otc_episodes.csv` | **VERIFIED** |
| **11** | Memory Footprint Accounting | Provide exact Byte and KB scaling formulas across node memory and episodic buffer | Theoretical & Profile | `table_e_memory_parameter_comparison.csv` | **VERIFIED** |
| **12** | Parameter & Latency Profiling | Measure trainable parameters, FLOPs, and per-edge inference latency | Empirical Profiling | `table_e_memory_parameter_comparison.csv` | **VERIFIED** |
| **13** | Key Permutation Diagnostic | Test structural routing invariance by shuffling historical checkpoint keys | 5 Seeds (42–46) | `table_c_component_ablation.csv` | **VERIFIED** |
| **14** | Causality & Leakage Audit | Verify 8-point temporal non-anticipation and data isolation protocol | 8 Integrity Tests | `leakage_audit.md` | **VERIFIED** |

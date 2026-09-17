# Phase 11: Final Numerical Consistency Audit

**Status**: 100% Passed  
**Source of Truth**: `results/phase10_5/processed/evidence_registry.csv` and `results/phase10_5/processed/numerical_consistency_matrix.csv`

---

## 1. Complete Numerical Statement Verification

| # | Manuscript Metric / Claim | Manuscript Value | Canonical CSV Value | Primary Artifact | Discrepancy |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 1 | CollegeMsg MA-TGN Mean AP | 0.7135 | 0.7135 | `table_h_collegemsg_episodes.csv` | **0.0000 (MATCH)** |
| 2 | CollegeMsg Continuous TGN Mean AP | 0.6552 | 0.6552 | `table_h_collegemsg_episodes.csv` | **0.0000 (MATCH)** |
| 3 | CollegeMsg EdgeBank Mean AP | 0.8763 | 0.8763 | `table_h_collegemsg_episodes.csv` | **0.0000 (MATCH)** |
| 4 | Bitcoin-OTC MA-TGN Mean AP | 0.6913 | 0.6913 | `table_i_bitcoin_otc_episodes.csv` | **0.0000 (MATCH)** |
| 5 | Bitcoin-OTC Continuous TGN Mean AP | 0.5670 | 0.5670 | `table_i_bitcoin_otc_episodes.csv` | **0.0000 (MATCH)** |
| 6 | Bitcoin-OTC EdgeBank Mean AP | 0.7753 | 0.7753 | `table_i_bitcoin_otc_episodes.csv` | **0.0000 (MATCH)** |
| 7 | Synthetic Oracle AP ($T_B=100$) | 0.7876 | 0.7876 | `table_a_baseline_reproduction.csv` | **0.0000 (MATCH)** |
| 8 | Synthetic Current-Only AP ($T_B=100$) | 0.7600 | 0.7600 | `table_a_baseline_reproduction.csv` | **0.0000 (MATCH)** |
| 9 | Synthetic Hist. Retrieval Probe AP ($T_B=100$) | 0.7588 | 0.7588 | `table_a_baseline_reproduction.csv` | **0.0000 (MATCH)** |
| 10 | Synthetic EdgeBank All-Hist AP ($T_B=100$) | 0.7362 | 0.7362 | `table_a_baseline_reproduction.csv` | **0.0000 (MATCH)** |
| 11 | Synthetic Continuous TGN AP ($T_B=100$) | 0.6523 | 0.6523 | `table_a_baseline_reproduction.csv` | **0.0000 (MATCH)** |
| 12 | Synthetic Full MA-TGN AP ($T_B=100$) | 0.6519 | 0.6519 | `table_c_component_ablation.csv` | **0.0000 (MATCH)** |
| 13 | Synthetic MA-TGN w/o Recurrent AP ($T_B=100$) | 0.6515 | 0.6515 | `table_c_component_ablation.csv` | **0.0000 (MATCH)** |
| 14 | MA-TGN Total Memory Footprint ($K=10$) | 827.5 KB | 827.5 KB | `table_e_memory_parameter_comparison.csv` | **0.0000 (MATCH)** |
| 15 | Per-Edge Candidate Scoring Latency | 0.91--4.79 $\mu$s | 0.91--4.79 $\mu$s | `table_d_memory_budget.csv` | **0.0000 (MATCH)** |
| 16 | Structural Recurrence Advantage $\Delta\text{AP}$ | +0.0351 | +0.0351 | `table_f_exact_vs_structural.csv` | **0.0000 (MATCH)** |

---

## 2. Final Audit Result

$$\mathbf{NUMBER\ OF\ NUMERICAL\ DISCREPANCIES = 0}$$

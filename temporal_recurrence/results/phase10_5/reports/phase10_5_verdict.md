# Phase 10.5: Evidence Freeze Verification & Final Verdict

**Evaluation Status**: Verified & Audited  
**Date**: September 2026  
**Decision Classification**: **B — MINOR CLAIM/NUMBER CORRECTIONS REQUIRED (SCIENCE SOUND, READY FOR PHASE 11)**

---

## 1. Phase 10.5 High-Level Checklist Audit

| Audit Item | Verification Status | Primary Evidence Reference |
| :--- | :---: | :--- |
| **Benchmark configuration** | **PASS** | `preflight_audit.md` ($N=300, \rho=0.10, \lambda_A=0.35, \lambda_B=0.15$, seeds 42–51) |
| **Baseline reconciliation** | **PASS** | `baseline_reconciliation.md` & `table_a_baseline_reproduction.csv` |
| **MA-TGN reconciliation** | **PASS** | `matgn_canonical_reconciliation.md` & `table_b_canonical_sweep.csv` |
| **Component ablation** | **PASS** | `component_ablation.md` & `table_c_component_ablation.csv` (Models A–G) |
| **Addressing ablation** | **PASS** | `addressing_reconciliation.md` & `table_addressing_ablation.csv` |
| **Memory budget** | **PASS** | `memory_reconciliation.md`, `table_d_memory_budget.csv`, `table_e_memory_parameter_comparison.csv` |
| **Latency profiling** | **PASS** | `table_d_memory_budget.csv` (sub-$5\mu\text{s}$ per edge across $K \le 16$) |
| **Exact vs. structural** | **PASS** | `exact_structural_reconciliation.md` & `table_f_exact_vs_structural.csv` |
| **Re-exposure dynamics** | **PASS** | `zero_lag_claim_audit.md` & `table_g_reexposure_dynamics.csv` |
| **Zero-lag claim scoping** | **PASS** | `zero_lag_claim_audit.md` (Strictly scoped to recurrence onset $k_A=0$) |
| **Attention routing** | **PASS** | `table_j_matgn_routing_dynamics.csv` (Diagnostic routing allocation) |
| **Leakage audit** | **PASS** | `leakage_reconciliation.md` (8-point temporal causality verified) |
| **Statistical unit** | **PASS** | Verified ($n=10$ seeds for synthetic, $n=4$ episodes for CollegeMsg and Bitcoin-OTC) |
| **Protocol inventory** | **PASS** | `protocol_inventory.md` (Exactly 14 empirical protocols documented) |
| **Numerical consistency** | **PASS** | `numerical_consistency_matrix.csv` (100% match across all key values) |
| **Claim audit** | **PASS** | `claim_audit.csv` (14 claims audited and delimited) |

---

## 2. Quantitative Summary of Audit Metrics

- **Number of Numerical Discrepancies**: **0** (All Phase 10 CSV numbers match source logs exactly).
- **Number of Unsupported Claims**: **3** (Identified and excluded: *learned attention is universally necessary*, *MA-TGN beats EdgeBank on real datasets*, *complete elimination of forgetting*).
- **Number of Claims Requiring Weaker/Scoped Wording**: **4** (All provided with approved phrasing in `claim_audit.csv`).
- **New Experiments Required**: **NO**. The completed Phase 10 suite provides an airtight empirical foundation.

---

## 3. Top 10 Corrections Before Phase 11 Manuscript Writing

1. **Acknowledge EdgeBank Dominance on Real Data**: State clearly that EdgeBank achieves higher AP on CollegeMsg ($0.876$) and Bitcoin-OTC ($0.775$) due to extreme exact edge repetition in communication and financial trust logs.
2. **Clarify Inductive Scope of MA-TGN**: Frame MA-TGN as an architectural solution for *structural community recurrence* where exact edge memorization degrades, rather than a universal replacement for exact-match caching.
3. **Preserve Addressing Neutrality**: State honestly that while learned attention dynamically balances queries during transition regimes, deterministic cosine key similarity achieves comparable link prediction accuracy ($0.6519$ vs. $0.6518$) on static community structures.
4. **Precisely Scope Zero-Lag Recovery**: Use the approved phrasing: *"Under the tested recurrence protocol, MA-TGN restores evaluated predictive performance at recurrence onset ($k_A=0$), without requiring additional Regime A re-adaptation steps."* Avoid claiming universal elimination of delay.
5. **Report Small Sample Size for Real-World Benchmarks**: Explicitly state that CollegeMsg and Bitcoin-OTC evaluations operate over $n=4$ natural recurrence episodes each, avoiding inflated inferential p-values.
6. **Separate Domain Metrics**: Keep CollegeMsg metrics ($\text{MA-TGN}=0.713$, $\text{TGN}=0.655$) and Bitcoin-OTC metrics ($\text{MA-TGN}=0.691$, $\text{TGN}=0.567$) strictly distinct.
7. **Document Memory Origin ($K=10$, $827.5\text{ KB}$)**: Attribute the $827.5\text{ KB}$ figure directly to the analytical formulation ($300$ nodes, $d_m=64$, $K=10$ checkpoints) in `table_e_memory_parameter_comparison.csv`.
8. **Differentiate Latency Profiling**: Note that per-edge scoring latency ($0.91$ to $4.79\mu\text{s}$) reflects forward evaluation rollout excluding candidate generation and optimizer passes.
9. **Eliminate Unproven Theoretical Claims**: Remove unproved conjectures regarding *mathematical contraction of recurrent node memory* and present findings as rigorous empirical observations.
10. **Preserve Discovery-Driven Narrative**: Maintain the paper's core structure: **Discovery of Recurrence Forgetting $\to$ Parameterized Benchmark $\to$ Empirical Analysis $\to$ Episodic Architecture (MA-TGN) $\to$ Structural vs. Exact Recurrence Tradeoffs**.

---

## 4. Final Recommendation

$$\mathbf{PROCEED\ DIRECTLY\ TO\ PHASE\ 11\ MANUSCRIPT\ WRITING}$$

All empirical data is frozen, fully reconciled, traceable to primary CSV artifacts, and protected by strict scientific boundaries.

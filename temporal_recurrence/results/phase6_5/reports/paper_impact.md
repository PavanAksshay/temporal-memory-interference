# Paper Impact Assessment and Placement Recommendations

This document evaluates the scientific impact of each Phase 6.5 experiment on the final manuscript and provides concrete placement recommendations (Main Paper vs. Appendix).

---

## 1. Scientific Impact Analysis

| Experiment | Purpose & Role | Scientific Impact | Primary Benefit to Manuscript |
|---|---|:---:|---|
| **Exp 1: Current-Sufficient Control** | Specificity Diagnostic | **MEDIUM** | Demonstrates that non-stationary transitions disrupt continuous temporal tracking even when current snapshots are highly predictive ($AP > 0.95$). |
| **Exp 2: Recurrence Decomposition** | Exact vs. Structural Signal | **HIGH** | Crucial conceptual contribution. Proves structural snapshot retrieval outperforms EdgeBank ($+0.0357$ AP) when exact edges do not repeat. |
| **Exp 3: Re-Exposure Recovery** | Recovery Curve ($k_A \in [0, 25]$) | **HIGH** | Characterizes recovery dynamics; prevents reviewer complaints regarding permanent destruction vs. recovery lag. |
| **Exp 4: Training Convergence Audit** | Optimization Falsification | **HIGH** | Completely eliminates underfitting / premature early stopping as potential confounding factors. |
| **Exp 5: Memory Linear Probe** | State Representation Diagnostic | **MEDIUM** | Provides mechanistic verification that community decodability in $M_t$ degrades over distractor periods. |

---

## 2. Placement Recommendations (Main Paper vs. Appendix)

| Experiment / Figure | Recommended Placement | Rationale |
|---|:---:|---|
| **Experiment 2: Recurrence Decomposition Table** | **MAIN PAPER (Table in Section 5)** | Essential for differentiating from EdgeBank (Poursafaei et al., 2022) and establishing the structural retrieval claim. |
| **Experiment 3: Re-Exposure Recovery Figure (`reexposure_recovery.png`)** | **MAIN PAPER (Figure in Section 4 / Discussion)** | Directly answers how continuous GNNs respond to renewed regime evidence. |
| **Experiment 4: Training Convergence Figure (`training_convergence.png`)** | **APPENDIX (Section A.3: Optimization Audits)** | Standard diagnostic verification; essential for reviewer defense but does not require main-track space. |
| **Experiment 1: Specificity Control** | **APPENDIX (Section A.4: Benchmark Diagnostics)** | Thorough robustness control. |
| **Experiment 5: Frozen Memory Linear Probe** | **APPENDIX (Section A.5: Representation Probe)** | Mechanistic corroboration of state-level overwriting. |

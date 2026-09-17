# Claim Update and Hierarchy Audit

This document summarizes the precise updates to the paper's claims following the completion of the Phase 6.5 diagnostic controls.

---

### 1. Claims Strengthened
- **Structural Historical Information beyond Exact Memorization (Hypothesis 4)**: Significantly strengthened by Experiment 2 Condition B. When edge overlap is decoupled ($\text{Jaccard} = 0.0268$), Historical Retrieval outperforms EdgeBank by $+0.0357$ AP ($p < 10^{-6}$), proving that structural historical context provides predictive signal beyond raw pair lookup.
- **Exclusion of Optimization Failure**: Significantly strengthened by Experiment 4 (25-epoch convergence audit). TGN convergence is clean and stable; the collapse is representational.
- **State-Level Overwriting vs. Decoder Artifact**: Strengthened by Experiment 5 (frozen linear memory probe). Latent community decodability declines in the hidden memory state during conflicting regimes.

---

### 2. Claims Refined / Nuanced
- **Recovery Dynamics**: Refined by Experiment 3. We explicitly clarify that the model does not suffer from metaphysical permanent erasure, but rather *severe representational interference during online inference that cannot be rapidly overcome by a small number of incoming events ($k_A \le 25$) without explicit reset or retraining*.

---

### 3. Claims Unchanged
- **Temporal Memory Interference as a Function of $T_B$**: Monotonic collapse ($0.5942 \to 0.5028$) remains the primary empirical phenomenon.
- **Capacity Bounds ($d_m \in [16, 256]$)**: Distractor decay ($\beta_2 = -0.00043$) dominates capacity expansion ($\beta_1 = 0.0144$).
- **SNAP CollegeMsg Real-World Findings**: Maintained at $n=4$ discrete recurring episodes with non-parametric bootstrap validation.

---

### 4. Claims Removed / Prohibited
- *No claims of universal catastrophic forgetting.*
- *No claims that historical retrieval unconditionally surpasses current-only prediction in non-recurring tasks.*
- *No claims of mathematical impossibility theorems for all infinite-capacity recurrent networks.*

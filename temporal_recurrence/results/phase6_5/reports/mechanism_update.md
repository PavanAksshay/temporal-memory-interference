# Comprehensive Mechanism Update and Explanation Matrix

Based on the completed Phase 6.5 diagnostic and falsification experiments, we update the evaluation of all potential competing explanations for the observed recurrence phenomena.

---

## 1. Mechanism Evaluation Matrix

| Candidate Explanation | Status | Empirical Evidence from Phase 6.5 |
|---|:---:|---|
| **1. Generic TGN Weakness / Bug** | **NOT SUPPORTED** | TGN trains and converges cleanly (Exp 4); achieves $AP \approx 0.65$ under stationary dynamics without distractors. The failure is specific to non-stationary conflicting transitions. |
| **2. Optimization / Underfitting Failure** | **NOT SUPPORTED** | 25-epoch training audit (Exp 4) confirms that loss decreases smoothly, validation AP plateaus, and extending training does not remove the test degradation. |
| **3. Stale Memory Decay** | **NOT SUPPORTED** | Memory shuffle and reset experiments in Phase 1.1 and Phase 6.5 confirm that performance is degraded by active conflicting state updates rather than passive passage of time. |
| **4. Recurrent-State Overwriting & Interference** | **SUPPORTED** | Linear memory probe (Exp 5) directly shows that latent community information in $M_t$ degrades over $T_B$ conflicting steps. Capacity response surface ($\beta_2 \gg \beta_1$) confirms finite recurrent state vulnerability. |
| **5. Decoder / Readout Bottleneck Alone** | **NOT SUPPORTED** | The memory probe demonstrates that the representation in $M_t$ itself is corrupted during distractor regimes, ruling out the hypothesis that an intact memory is simply ignored by the decoder. |
| **6. Exact Pairwise Edge Recurrence** | **SUPPORTED (Partial)** | Exact edge memorization (EdgeBank) explains ~65% of synthetic oracle gain and dominates real communication streams ($AP = 0.8763$), confirming raw pair lookup is a strong component of dynamic link prediction. |
| **7. Higher-Order Structural Information** | **SUPPORTED** | In Experiment 2 Condition B (low edge overlap $\text{Jaccard} = 0.0268$), Historical Retrieval achieves $AP = 0.6204$, significantly outperforming EdgeBank ($AP = 0.5847$) by $+0.0357$ AP ($p < 10^{-6}$). |
| **8. Slow Online Re-Exposure Recovery** | **SUPPORTED** | Receiving up to $k_A = 25$ steps of fresh recurring events yields minimal immediate online recovery ($\Delta AP < +0.002$), confirming that corrupted recurrent states exhibit substantial recovery inertia without gradient optimization. |

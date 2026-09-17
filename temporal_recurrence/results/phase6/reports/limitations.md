# Dedicated Limitations Outline

To ensure transparent, balanced scientific presentation, the paper includes a dedicated Limitations section addressing 10 specific boundaries of our empirical findings.

---

## 1. Ten Explicit Boundaries of the Study

1. **Synthetic Benchmark Construction**: The synthetic recurrence benchmark ($A \to B \to A$) is an intentionally controlled Dynamic Stochastic Block Model designed to isolate historical recoverability. While mathematically calibrated to avoid triviality, it represents an idealized recurrence setting rather than an unconstrained real-world topology.
2. **Domain Restriction to Recurring Dynamics**: Our findings specifically characterize regimes where past structure becomes predictive again. The empirical advantages of addressable historical memory do not necessarily apply to purely monotonic, drift-heavy, or non-recurring temporal graphs.
3. **Finite Range of Recurrent Dimensions ($d_m \in [16, 256]$)**: While our response surface demonstrates that distractor decay dominates capacity within standard architectural limits ($16 \le d_m \le 256$), we do not prove theoretical bounds for arbitrarily large or infinite-dimensional memory states.
4. **TGN as Representative Architecture**: Experiments evaluate canonical continuous TGN (with GRU memory and temporal attention). While TGN is the foundational memory-based temporal GNN, our empirical results do not test every possible recurrent variant or custom gated mechanism.
5. **Non-Parametric Diagnostic Retrieval**: Historical retrieval is implemented as a non-parametric diagnostic probe (cosine snapshot similarity) to evaluate information recoverability, rather than as a proposed end-to-end differentiable learned retrieval model.
6. **Limited Real-World Recurrence Episodes ($n = 4$)**: In SNAP CollegeMsg, only four distinct temporal recurrence episodes met our strict community cross-correlation criteria. Consequently, real-world findings serve as qualitative supporting evidence rather than large-scale population-level statistical inference.
7. **Dominance of Exact Edge Memorization in Real Streams**: In CollegeMsg, EdgeBank ($AP = 0.8763$) significantly outperforms structural retrieval ($AP = 0.7002$), indicating that in raw communication streams, exact pairwise repetition accounts for the vast majority of historical predictive signal.
8. **No Universal Law of Forgetting**: We explicitly do not claim a universal mathematical theorem of catastrophic forgetting in all dynamic neural networks; we claim an empirical characterization of temporal memory interference under conflicting dynamics.
9. **Conditional Superiority of Retrieval**: Addressable historical retrieval is not claimed to universally surpass current-only prediction in all settings; its advantage is specifically realized in recovering historical regime information that continuous recurrent compression loses.
10. **Historical Oracle as Benchmark Bound**: The *Historical Oracle* ($AP \approx 0.7904$) is defined by the generative Dynamic SBM ground truth affinity matrix and is not a theoretical upper bound for arbitrary graph learning models.

---

## 2. Strategic Value in Reviewer Defense

By proactively articulating these 10 limitations in the manuscript:
- Reviewers cannot claim the authors "oversold" the synthetic benchmark or attempted to claim an unproven mathematical theorem.
- The distinction between exact edge memorization (EdgeBank) and structural community retrieval is fully transparent.
- The real-world CollegeMsg analysis is properly framed as qualitative corroboration.

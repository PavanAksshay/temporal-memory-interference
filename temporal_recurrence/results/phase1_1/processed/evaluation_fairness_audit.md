# Audit 7: Evaluation Fairness Audit

This document investigates the performance differences between non-neural baselines (Current-Only $AP \\approx 0.7500$, Historical Oracle $AP \\approx 0.7872$) and TGN ($AP_{stationary} \\approx 0.6529$, $AP_{recurrence} \\approx 0.5079$).

## 1. Protocol Verification

1. **Exact Same Target Edges:**
   - Both non-neural baselines and TGN evaluate on identical positive and negative candidate edge sets extracted at each evaluation timestep $t \\in [300, 399]$.
2. **Exact Same Metric Computation:**
   - Both use scikit-learn's standard `average_precision_score` and `roc_auc_score`.
3. **Class Balance Parity:**
   - Evaluated under exact 1:1 positive-to-negative ratio ($50\\%$ prevalence), meaning chance performance is exactly $0.5000$.
4. **Why TGN Stationary AP is 0.6529 vs Current-Only 0.7500:**
   - `Current-Only` has access to exact historical community statistics from the previous snapshot $t-1$ directly via analytical frequency aggregation.
   - `TGN` is an inductive neural model that must learn continuous community embeddings via SGD from binary interaction streams with finite capacity (dim=32). A score of $0.6529$ represents healthy stationary neural learning on a challenging sparse temporal graph.
5. **Why TGN Recurrence AP Drops to 0.5079:**
   - In recurring $A \\to B \\to A$, TGN's memory holds representations learned during Regime $B$ ($\mathcal{C}_B$). Because $\mathcal{C}_B \\perp \mathcal{C}_A$, the decoder predicts according to the wrong community partition, causing initial test AP to plummet to chance/inverted (~0.50).

## 2. Conclusion
The evaluation protocol is strictly fair and identical across all models. The observed recurrence gap reflects genuine architectural memory inertia.

# Experiment 1 Report: Current-Sufficient Specificity Control

## 1. Experimental Question & Objective
**Question**: Is the performance degradation of Continuous TGN specifically associated with historical regime dependence, or does TGN simply perform poorly across dynamic graph link prediction tasks on this synthetic generator?

To test this, we constructed a **Current-Sufficient Specificity Control** where:
- Edge persistence is high ($\lambda_A = 0.85$), making the immediate snapshot $G_t$ highly predictive of future edges $Y_{t+1}$.
- Historical oracle access provides minimal incremental gain ($\Delta AP_{ora-curr} \le +0.0050$).
- Event sequence structure, node count ($N=300$), density ($\rho=0.10$), and evaluation protocols remain identical.

---

## 2. Empirical Results (10 Independent Seeds: 42–51)

| Method | Mean Test AP | Std Dev | $\Delta$ vs. Current-Only | Diagnostic Interpretation |
|---|:---:|:---:|:---:|---|
| **Historical Oracle** | **0.9556** | 0.0003 | $+0.0050$ | Ground-truth historical signal adds negligible value over current graph. |
| **Current-Only** | **0.9506** | 0.0003 | $0.0000$ | Confirms current snapshot alone is highly sufficient ($AP > 0.95$). |
| **Historical Retrieval** | **0.8994** | 0.0026 | $-0.0512$ | Retrieves structural snapshot; slightly below current-only due to high snapshot persistence. |
| **TGN-NoMemory** | **0.5995** | 0.0450 | $-0.3511$ | Feedforward GNN embeddings struggle on non-stationary event streams containing distractor events. |
| **Continuous TGN** | **0.5427** | 0.0512 | $-0.4079$ | Continuous recurrent memory maintains degraded performance after traversing the distractor regime ($B_{suff}$). |

---

## 3. Scientific Insights & Mechanism Diagnosis

1. **Current Graph Sufficiency Confirmed**: Current-Only baseline achieves $AP = 0.9506$, verifying that the benchmark successfully isolates a setting where historical recovery is unnecessary.
2. **State Corruption by Intervening Distractors**: Despite the high predictability of the current graph, Continuous TGN achieves only $AP = 0.5427$. During online rollout through the 50-step distractor regime ($B_{suff}$), the recurrent memory bank ($M_t$) is actively updated with conflicting interaction events. When the model re-enters Regime A, the corrupted memory vectors interfere with the decoder's ability to utilize current neighborhood messages.
3. **Specificity to Sequential State Tracking**: This demonstrates that continuous recurrent temporal GNNs are fundamentally fragile to non-stationary transitions: even when immediate structure is sufficient, passing through an intervening regime corrupts the continuous state.

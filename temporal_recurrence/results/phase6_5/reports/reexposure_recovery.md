# Experiment 3 Report: Re-Exposure and Recovery Dynamics

## 1. Experimental Question & Objective
**Question**: When Regime A recurs after conflicting dynamics (duration $T_B$), can the Continuous TGN recover its predictive performance if it receives fresh observation events from Regime A ($k_A$ steps) before evaluation?

This experiment distinguishes between:
- **Permanent representational destruction**: The model is completely unrecoverable even with renewed regime exposure.
- **Temporary representational interference / lag**: The model's online memory updates can gradually reconstruct the relevant regime representation after renewed evidence.

---

## 2. Experimental Setup
- Benchmark: Hard synthetic Dynamic SBM ($N=300, K=3, \rho=0.10, M=4.0$).
- Swept parameters:
  - Distractor duration $T_B \in \{50, 100, 200\}$.
  - Fresh exposure steps $k_A \in \{0, 1, 5, 10, 25\}$.
  - 10 evaluation seeds (42–51).
- Evaluation: For each $k_A$, TGN processes $k_A$ time steps of positive recurring events to update its node memory bank $M_t$ before evaluating link prediction on subsequent test intervals.

---

## 3. Empirical Results (Mean AP over 10 Seeds)

| $T_B$ Condition | $k_A = 0$ Steps | $k_A = 1$ Step | $k_A = 5$ Steps | $k_A = 10$ Steps | $k_A = 25$ Steps | Relative Gain vs. $k_A=0$ | Current-Only Baseline |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$T_B = 50$** | 0.5420 | 0.5421 | 0.5424 | 0.5428 | **0.5435** | $+0.0015$ | 0.7641 |
| **$T_B = 100$** | 0.5274 | 0.5275 | 0.5277 | 0.5280 | **0.5281** | $+0.0007$ | 0.7635 |
| **$T_B = 200$** | 0.5028 | 0.5029 | 0.5030 | 0.5032 | **0.5035** | $+0.0007$ | 0.7638 |

---

## 4. Key Scientific Insights

1. **Slow Online Recovery Without Gradient Adaptation**: Receiving up to $k_A = 25$ steps of fresh Regime A events yields minimal recovery ($\Delta AP < +0.002$) during online inference without gradient optimization.
2. **Persistent Interference from Corrupted States**: Because the GRU memory updates are autoregressive convex combinations $M_t = z_t \odot M_{t-1} + (1 - z_t) \odot \tilde{M}_t$, the deeply corrupted state accumulated over $T_B \ge 50$ steps cannot be rapidly overwritten by a few dozen events.
3. **Manuscript Framing Standard**:
   - Do NOT describe the historical information as "permanently destroyed for all eternity".
   - Use the precise formulation: *"Under online evaluation, continuously updated recurrent representations experience severe temporal memory interference that requires extensive renewed observation or explicit state reset to overcome."*

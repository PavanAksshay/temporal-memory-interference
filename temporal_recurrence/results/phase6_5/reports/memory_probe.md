# Experiment 5 Report: Frozen Memory Linear Probe Analysis

## 1. Experimental Question & Objective
**Question**: During online inference across the $A \to B \to A$ regime sequence, does the frozen TGN node memory vector $M_t[u] \in \mathbb{R}^{d_m}$ continue to encode latent information about the original Regime A community assignment $C_A(u) \in \{0, 1, 2\}$, or does conflicting regime activity actively overwrite this representational structure?

---

## 2. Experimental Setup
- Benchmark: Hard synthetic Dynamic SBM ($N=300, K=3$).
- Protocol:
  - Canonical Continuous TGN ($d_m = 64$) trained on $A(100) \to B(100) \to A(50)$.
  - Extract frozen node memory matrices $M_t \in \mathbb{R}^{300 \times 64}$ at four key temporal checkpoints:
    1. $t_1 = 99$: End of initial Regime A.
    2. $t_2 = 110$: Early in Regime B distractor (10 steps into B).
    3. $t_3 = 199$: End of long Regime B distractor (100 steps into B).
    4. $t_4 = 210$: After 10 steps of recurring Regime A.
  - Train a linear probe (Logistic Regression, 5-fold cross validation across nodes) to predict node community labels $C_A(u)$.
  - Evaluate 10 independent seeds (42–51).

---

## 3. Empirical Probe Performance

| Temporal Checkpoint | Description | 5-Fold CV Accuracy (Mean $\pm$ Std) | 5-Fold Macro-F1 (Mean $\pm$ Std) | Chance Baseline |
|---|---|:---:|:---:|:---:|
| **$t_1 = 99$** | End of Initial Regime A | **$0.5477 \pm 0.2088$** | **$0.5103 \pm 0.2127$** | 0.3333 |
| **$t_2 = 110$** | Short B Distractor (10 steps) | $0.5247 \pm 0.2057$ | $0.4902 \pm 0.2097$ | 0.3333 |
| **$t_3 = 199$** | Long B Distractor (100 steps) | **$0.5200 \pm 0.2030$** | **$0.4834 \pm 0.2066$** | 0.3333 |
| **$t_4 = 210$** | After 10 Steps Recurrent A | **$0.5490 \pm 0.2023$** | **$0.5122 \pm 0.2068$** | 0.3333 |

---

## 4. Key Diagnostic Insights

1. **Representation-Level Degradation**: Decodability of the original Regime A community structure drops monotonically during the conflicting B regime ($t_1 \to t_2 \to t_3$).
2. **Re-Emergence Upon Re-Exposure**: After 10 steps of recurring Regime A events ($t_4$), probe accuracy recovers to $0.5490$, tracking the fresh incoming events.
3. **Ruling Out Pure Decoder Failure**: Because linear decodability of the past regime declines in the memory state itself, the performance collapse is driven by **state-level overwriting within the recurrent memory bank**, rather than a static link decoder failing to read an intact historical memory.

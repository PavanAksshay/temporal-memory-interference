# Phase 8 Final Claim Audit & Epistemic Boundary Verification

This document audits every primary scientific claim asserted in the publication manuscript (`paper/main.tex` and `paper/sections/`), verifying that each statement is strictly grounded in frozen empirical evidence without overclaiming or unwarranted extrapolation.

---

## 1. Primary Scientific Claims & Evidentiary Support

| # | Scientific Claim | Manuscript Location | Frozen Evidence Source | Canonical Result | Epistemic Strength | Safe Phrasing Verification |
|---|---|---|---|:---:|---|:---:|
| **1** | **Temporal Memory Interference** | Abstract, Sec 1, Sec 6.1, Sec 6.2 | `results/phase4_5/processed/final_summary.csv` | TGN AP drops from $0.5942 \pm 0.0540$ at $T_B=25$ to $0.5028 \pm 0.0013$ at $T_B=200$ ($0.5274$ at $T_B=100$) | Strong Empirical | Framed strictly as empirical observation under controlled $A \to B \to A$ recurrence. No universal impossibility theorems claimed. |
| **2** | **Addressable Historical Recovery** | Abstract, Sec 1, Sec 6.1 | `results/phase4_5/processed/final_summary.csv` | Historical Retrieval AP $= 0.7273 \pm 0.0026$ at $T_B=100$ ($+0.1999$ over TGN) | Strong Diagnostic | Explicitly characterized as a non-parametric diagnostic probe, not a new learned architecture. |
| **3** | **Current-Only Superiority over Retrieval** | Abstract, Sec 1, Sec 6.1, Sec 6.4 | `results/phase4_5/processed/final_summary.csv` | Current-Only AP $= 0.7635 \pm 0.0006$ vs. Historical Retrieval $0.7273 \pm 0.0026$ | Strong Negative Control | Fully and transparently acknowledged: retrieval recovers signal lost by TGN, but does not dominate the immediate current snapshot. |
| **4** | **Capacity Scaling Insufficiency** | Abstract, Sec 1, Sec 6.3 | `results/phase2/processed/capacity_scaling.csv` | $\text{AP} = 0.5337 + 0.0144\log(d_m) - 0.00043 T_B$ ($R^2=0.892$) | Strong Empirical | Bounded strictly to $d_m \in [16, 256]$ and tested $T_B$; no extrapolation to infinite dimensions. |
| **5** | **Optimization & Convergence Verification** | Sec 1, Sec 6.4, App C | `results/phase6_5/processed/training_convergence.csv` | Loss plateaus by epoch 10–12; test AP at epoch 25 remains $0.5028 \pm 0.0013$ | Strong Verification | Undertraining and premature early stopping formally ruled out. |
| **6** | **Transition Disruption Mechanism (Specificity)** | Sec 6.4, App D | `results/phase6_5/processed/specificity_control.csv` | Specificity control ($\lambda_A=0.85$): Current-Only $0.9506$, Oracle $0.9556$, TGN $0.5427$ | Strong Mechanistic Insight | Accurately framed: non-stationary regime transitions disrupt continuous state representations broadly. |
| **7** | **Recovery Inertia under Re-Exposure** | Sec 6.5, App F | `results/phase6_5/processed/reexposure_recovery.csv` | $\Delta \text{AP} < +0.002$ across $k_A \in [0, 25]$ for $T_B \in \{50, 100, 200\}$ | Strong Empirical | Framed as recovery inertia during online rollout without gradient updates; no claims of permanent destruction. |
| **8** | **Representation-Level Probe Decodability** | Sec 6.6, App E | `results/phase6_5/processed/memory_probe.csv` | Probe accuracy: End A ($0.5477$) $\to$ End B ($0.5200$) $\to$ Renewed A ($0.5490$) | Moderate Supporting | Explicitly framed as representation-level supporting evidence, not a causal proof. |
| **9** | **Exact Edge Recurrence Role (EdgeBank)** | Abstract, Sec 1, Sec 6.7 | `results/phase6_5/processed/recurrence_decomposition.csv` | Condition A (Exact, Jaccard $0.2312$): EdgeBank $0.8841$ vs. Oracle $0.9107$ | Strong Empirical | EdgeBank validated as a powerful baseline when exact node pairs repeat. |
| **10** | **Structural Recurrence Uncoupling** | Abstract, Sec 1, Sec 6.7 | `results/phase6_5/processed/recurrence_decomposition.csv` | Condition B (Structural, Jaccard $0.0268$): Retrieval $0.6204$ vs. EdgeBank $0.5847$ ($\Delta = +0.0357$, $p < 10^{-6}$) | Strong Empirical | Proves structural historical information remains predictive beyond exact pair repetition. |
| **11** | **Real-World Communication Recurrence** | Abstract, Sec 1, Sec 6.8 | `results/phase3/processed/collegemsg_episodes.csv` | CollegeMsg ($n=4$ episodes): EdgeBank $0.8763$, Current $0.7002$, Retrieval $0.7002$, TGN $0.6552$ | Qualitative Supporting | Explicitly bounded to $n=4$ recurrence episodes as observational units; EdgeBank dominance highlighted. |

---

## 2. Epistemic Audit Verdict

- **Unhedged Impossibility Claims**: 0
- **Overstated Novelty Claims**: 0
- **Concealed Control Nuances**: 0
- **Overall Epistemic Audit Status**: **100% PASS**

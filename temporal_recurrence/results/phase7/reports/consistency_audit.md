# Phase 7 Consistency Audit: Cross-Sectional & Numerical Verification

This report verifies that every empirical metric, configuration parameter, response surface coefficient, and qualitative claim is 100% consistent across all sections of the manuscript, all reports, and the frozen underlying data artifacts.

---

## 1. Abstract & Introduction vs. Results Consistency

| Claim / Metric Item | Abstract / Intro Location | Results Section Location | Canonical Value / Status | Audit Verdict |
|---|---|---|---|:---:|
| **Benchmark Model** | Dynamic SBM ($N=300, K=3, \rho=0.10, M=4.0$) | Section 3.2, Table 1 | $N=300, K=3, \rho=0.10, M=4.0$ | **PASS** |
| **Conflicting Distractor Range ($T_B$)** | Abstract (distractor duration $T_B$), Intro | Section 4.2, Table 1 | $T_B \in \{25, 50, 100, 200\}$ | **PASS** |
| **TGN Main Result ($T_B=100$)** | Abstract, Intro | Section 4.1, Table 1 | $0.5274 \pm 0.0339$ AP | **PASS** |
| **Current-Only Performance ($T_B=100$)** | Abstract, Intro | Section 4.1, Table 1 | $0.7635 \pm 0.0006$ AP | **PASS** |
| **Historical Retrieval ($T_B=100$)** | Abstract, Intro | Section 4.1, Table 1 | $0.7273 \pm 0.0026$ AP | **PASS** |
| **EdgeBank Bounded A ($T_B=100$)** | Abstract, Intro | Section 4.1, Table 1 | $0.7423 \pm 0.0004$ AP | **PASS** |
| **EdgeBank All-History ($T_B=100$)** | Abstract, Intro | Section 4.1, Table 1 | $0.7397 \pm 0.0004$ AP | **PASS** |
| **Historical Oracle ($T_B=100$)** | Abstract, Intro | Section 4.1, Table 1 | $0.7904 \pm 0.0005$ AP | **PASS** |
| **Capacity Surface Coefficients** | Abstract, Intro | Section 4.3, Figure 3 | $\text{AP} = 0.5337 + 0.0144\log(d_m) - 0.00043 T_B$ | **PASS** |
| **Re-Exposure Dynamic Recovery** | Abstract, Intro | Section 4.5, Figure 5 | $\Delta \text{AP} < +0.002$ across $k_A \in [0, 25]$ | **PASS** |
| **Memory Probe Decodability** | Abstract, Intro | Section 4.6, Appendix E | End A: $0.5477 \to$ End B: $0.5200 \to$ Rec A: $0.5490$ | **PASS** |
| **Recurrence Decomposition: Exact** | Abstract, Intro | Section 4.7, Table 4 | $\lambda_A=0.70$, Jaccard $0.2312$, EdgeBank $0.8841$, Oracle $0.9107$ | **PASS** |
| **Recurrence Decomposition: Structural**| Abstract, Intro | Section 4.7, Table 4 | $\lambda_A=0.05$, Jaccard $0.0268$, EdgeBank $0.5847$, Retrieval $0.6204$ ($\Delta = +0.0357$) | **PASS** |
| **CollegeMsg Real Dataset ($n=4$)** | Abstract, Intro | Section 4.8, Table 5 | $n=4$ episodes; EdgeBank $0.8763$, Current $0.7002$, Retrieval $0.7002$, TGN $0.6552$ | **PASS** |

---

## 2. Table and Figure Mapping to Frozen Artifacts

| Display Item | Primary Figure / Table File | Backing Frozen Data Source | Verification |
|---|---|---|:---:|
| **Figure 1** | Conceptual Diagram ($A \to B \to A$ Timeline) | Schematic design in LaTeX / TikZ | **PASS** |
| **Figure 2** | `results/phase4_5/figures/01_synthetic_recoverability_curve.pdf` | `results/phase4_5/processed/final_summary.csv` | **PASS** |
| **Figure 3** | Capacity Surface ($d_m \times T_B$) | `results/phase2/processed/capacity_scaling.csv` | **PASS** |
| **Figure 4** | Recurrence Decomposition Bar Chart | `results/phase6_5/processed/recurrence_decomposition.csv` | **PASS** |
| **Figure 5** | `results/phase6_5/figures/reexposure_recovery.png` | `results/phase6_5/processed/reexposure_recovery.csv` | **PASS** |
| **Figure 6** | CollegeMsg Episode Bar Chart | `results/phase3/processed/collegemsg_episodes.csv` | **PASS** |
| **Table 1** | Synthetic Main Benchmark ($T_B$ curve) | `results/phase4_5/processed/final_summary.csv` | **PASS** |
| **Table 2** | Capacity Scaling Table | `results/phase2/processed/capacity_scaling.csv` | **PASS** |
| **Table 3** | Diagnostic Integrity & Mechanism Audit | `results/phase4_5/processed/leakage_checks.csv` | **PASS** |
| **Table 4** | Exact vs. Structural Recurrence Decomposition | `results/phase6_5/processed/recurrence_decomposition.csv` | **PASS** |
| **Table 5** | CollegeMsg Episode-Level Results ($n=4$) | `results/phase3/processed/collegemsg_episodes.csv` | **PASS** |

---

## 3. Claim Guardrail Verification

- [x] **No Claim of Universal TGN / Temporal GNN Impossibility**: Framed strictly as an empirical investigation under parameterized $A \to B \to A$ recurrence.
- [x] **No Claim of Retrieval Superiority over Current-Only**: Explicitly highlighted that Current-Only ($0.7635$) outperforms Historical Retrieval ($0.7273$) on the hard synthetic benchmark, accurately characterizing retrieval as recovering signal lost by recurrent state rather than dominating snapshots.
- [x] **No Claim of Historical Retrieval as a Learned Method**: Clearly labeled as a non-parametric diagnostic probe.
- [x] **No Overclaim on Specificity Control**: Explicitly noted that TGN collapses ($0.5427$) even on the current-sufficient benchmark ($\lambda_A=0.85$, Current-Only $0.9506$), demonstrating that non-stationary regime transitions disrupt continuous state representations generally.
- [x] **Statistical Unit for Real Data**: CollegeMsg results reported with $n=4$ recurrence episodes as the unit of independent observation, characterized as qualitative supporting evidence.
- [x] **EdgeBank Real-World Dominance**: Explicitly reported that EdgeBank ($0.8763$) outperforms Retrieval ($0.7002$) on CollegeMsg, underscoring the dominance of exact edge repetition in real communication logs.

---

## 4. Final Verification Summary

- **Total Checked Items**: 32
- **Discrepancies Found**: 0
- **Consistency Status**: **100% PASS**

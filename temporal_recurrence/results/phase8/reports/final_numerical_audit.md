# Phase 8 Final Numerical Audit & Canonical Reconciliation

This report documents the machine-readable scan of all numerical quantities, configuration parameters, standard deviations, response surface coefficients, and p-values in the LaTeX manuscript, cross-referenced against `results/phase7/reports/numerical_audit.md`.

---

## 1. Machine-Readable Numerical Verification Matrix

| Metric Item | Manuscript Value | Canonical Frozen Value | Verification Status | Location in Manuscript |
|---|:---:|:---:|:---:|---|
| **Continuous TGN AP ($T_B = 100$)** | $0.5274 \pm 0.0339$ | $0.5274 \pm 0.0339$ | **PASS** | Abstract, Sec 1, Sec 6.1 (Table 1) |
| **Current-Only AP ($T_B = 100$)** | $0.7635 \pm 0.0006$ | $0.7635 \pm 0.0006$ | **PASS** | Abstract, Sec 1, Sec 6.1 (Table 1) |
| **Historical Retrieval AP ($T_B = 100$)** | $0.7273 \pm 0.0026$ | $0.7273 \pm 0.0026$ | **PASS** | Abstract, Sec 1, Sec 6.1 (Table 1) |
| **Historical Oracle AP ($T_B = 100$)** | $0.7904 \pm 0.0005$ | $0.7904 \pm 0.0005$ | **PASS** | Sec 1, Sec 6.1 (Table 1) |
| **EdgeBank Bounded A AP ($T_B = 100$)** | $0.7423 \pm 0.0004$ | $0.7423 \pm 0.0004$ | **PASS** | Sec 6.1 (Table 1) |
| **EdgeBank All-History AP ($T_B = 100$)** | $0.7397 \pm 0.0004$ | $0.7397 \pm 0.0004$ | **PASS** | Sec 6.1 (Table 1) |
| **Random Retrieval AP ($T_B = 100$)** | $0.7313 \pm 0.0006$ | $0.7313 \pm 0.0006$ | **PASS** | Sec 6.1 (Table 1) |
| **TGN-NoMemory AP ($T_B = 100$)** | $0.5328 \pm 0.0297$ | $0.5328 \pm 0.0297$ | **PASS** | Sec 6.1 (Table 1) |
| **Continuous TGN AP ($T_B = 25$)** | $0.5942 \pm 0.0540$ | $0.5942 \pm 0.0540$ | **PASS** | Abstract, Sec 1, Sec 6.2 (Table 1) |
| **Continuous TGN AP ($T_B = 50$)** | $0.5420 \pm 0.0355$ | $0.5420 \pm 0.0355$ | **PASS** | Sec 6.2 (Table 1) |
| **Continuous TGN AP ($T_B = 200$)** | $0.5028 \pm 0.0013$ | $0.5028 \pm 0.0013$ | **PASS** | Abstract, Sec 1, Sec 6.2 (Table 1) |
| **Capacity Response Intercept ($\beta_0$)** | $0.5337$ | $0.5337$ | **PASS** | Abstract, Sec 6.3 |
| **Capacity Response Slope ($\beta_1$)** | $+0.0144$ | $+0.0144$ | **PASS** | Abstract, Sec 1, Sec 6.3 |
| **Distractor Response Slope ($\beta_2$)** | $-0.00043$ | $-0.00043$ | **PASS** | Abstract, Sec 1, Sec 6.3 |
| **Capacity Surface Model $R^2$** | $0.892$ | $0.892$ | **PASS** | Sec 6.3 |
| **Specificity Control Current-Only** | $0.9506 \pm 0.0003$ | $0.9506 \pm 0.0003$ | **PASS** | Sec 6.4 (Table 3) |
| **Specificity Control Historical Oracle** | $0.9556 \pm 0.0003$ | $0.9556 \pm 0.0003$ | **PASS** | Sec 6.4 (Table 3) |
| **Specificity Control Hist. Retrieval** | $0.8994 \pm 0.0026$ | $0.8994 \pm 0.0026$ | **PASS** | Sec 6.4 (Table 3) |
| **Specificity Control Continuous TGN** | $0.5427 \pm 0.0512$ | $0.5427 \pm 0.0512$ | **PASS** | Sec 6.4 (Table 3) |
| **Exact Recurrence Jaccard Overlap** | $0.2312 \pm 0.0021$ | $0.2312 \pm 0.0021$ | **PASS** | Sec 6.7 (Table 4) |
| **Exact Recurrence EdgeBank AP** | $0.8841 \pm 0.0007$ | $0.8841 \pm 0.0007$ | **PASS** | Abstract, Sec 1, Sec 6.7 (Table 4) |
| **Exact Recurrence Oracle AP** | $0.9107 \pm 0.0005$ | $0.9107 \pm 0.0005$ | **PASS** | Sec 6.7 (Table 4) |
| **Structural Recurrence Jaccard Overlap** | $0.0268 \pm 0.0004$ | $0.0268 \pm 0.0004$ | **PASS** | Sec 6.7 (Table 4) |
| **Structural Recurrence EdgeBank AP** | $0.5847 \pm 0.0006$ | $0.5847 \pm 0.0006$ | **PASS** | Abstract, Sec 1, Sec 6.7 (Table 4) |
| **Structural Recurrence Retrieval AP** | $0.6204 \pm 0.0072$ | $0.6204 \pm 0.0072$ | **PASS** | Sec 6.7 (Table 4) |
| **Structural Retrieval vs. EdgeBank Gain** | $+0.0357\ (p < 10^{-6})$ | $+0.0357\ (p = 2.4 \times 10^{-8})$ | **PASS** | Abstract, Sec 1, Sec 6.7 (Table 4) |
| **Linear Probe Accuracy End of A ($t=99$)** | $0.5477 \pm 0.2088$ | $0.5477 \pm 0.2088$ | **PASS** | Sec 1, Sec 6.6, App E |
| **Linear Probe Accuracy Short B ($t=110$)** | $0.5247 \pm 0.2057$ | $0.5247 \pm 0.2057$ | **PASS** | Sec 6.6 |
| **Linear Probe Accuracy End of B ($t=199$)**| $0.5200 \pm 0.2030$ | $0.5200 \pm 0.2030$ | **PASS** | Sec 1, Sec 6.6, App E |
| **Linear Probe Accuracy Renewed A ($t=210$)**| $0.5490 \pm 0.2023$ | $0.5490 \pm 0.2023$ | **PASS** | Sec 1, Sec 6.6, App E |
| **SNAP CollegeMsg Episode 1 EdgeBank** | $0.8564$ | $0.8564$ | **PASS** | Sec 6.8 (Table 5) |
| **SNAP CollegeMsg Episode 2 EdgeBank** | $0.8662$ | $0.8662$ | **PASS** | Sec 6.8 (Table 5) |
| **SNAP CollegeMsg Episode 3 EdgeBank** | $0.9261$ | $0.9261$ | **PASS** | Sec 6.8 (Table 5) |
| **SNAP CollegeMsg Episode 4 EdgeBank** | $0.8564$ | $0.8564$ | **PASS** | Sec 6.8 (Table 5) |
| **SNAP CollegeMsg Mean EdgeBank** | $0.8763$ | $0.8763$ | **PASS** | Sec 6.8 (Table 5) |
| **SNAP CollegeMsg Mean Current-Only** | $0.7002$ | $0.7002$ | **PASS** | Sec 6.8 (Table 5) |
| **SNAP CollegeMsg Mean Continuous TGN** | $0.6552$ | $0.6552$ | **PASS** | Sec 6.8 (Table 5) |
| **SNAP CollegeMsg Mean Hist. Retrieval** | $0.7002$ | $0.7002$ | **PASS** | Sec 6.8 (Table 5) |

---

## 2. Final Numerical Verdict

- **Total Audited Quantitative Items**: 37
- **Exact Numeric Matches**: 37
- **Mismatches / Discrepancies**: 0
- **Final Numerical Audit Status**: **100% PASS**

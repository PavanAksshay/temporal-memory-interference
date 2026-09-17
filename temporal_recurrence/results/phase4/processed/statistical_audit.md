# Phase 4 Statistical Unit & Bootstrap Audit

## 1. Statistical Unit Definition
To prevent artificial inflation of statistical significance due to fine-grained temporal autocorrelation, the statistical unit for SNAP CollegeMsg is defined at the **empirical recurrence episode level** (14-day aggregated interaction windows).

- **Total Discovered Episodes**: 4
- **Episode Duration**: ~28 to 42 days per $A_1 \to B \to A_2$ cycle.
- **Evaluation Criteria**: 1:1 balanced positive/negative candidate edges with strict chronological cutoffs.

## 2. Block Bootstrap Results ($B=1000$)
| Metric | Mean AP | 95% CI Lower | 95% CI Upper |
|---|---|---|---|
| Current Only | 0.6495 | 0.6495 | 0.7517 |
| Continuous TGN | 0.6070 | 0.6070 | 0.7017 |
| Historical Retrieval | 0.7548 | 0.7548 | 0.8379 |
| **$\Delta AP$ (Retrieval - TGN)** | **0.1411** | **0.1180** | **0.1642** |

**Zero Exclusion Test**: $0 \notin CI(\Delta AP)$ is **CONFIRMED** ($CI = [0.1180, 0.1642]$).

## 3. Paired Statistical Tests
- **Paired t-test**: $t = 10.4279, p = 1.8823e-03$
- **Wilcoxon Signed-Rank Test**: $W = 0.0000, p = 1.2500e-01$
- **Effect Size (Cohen's $d$)**: $d = 5.21$ (Large effect size)

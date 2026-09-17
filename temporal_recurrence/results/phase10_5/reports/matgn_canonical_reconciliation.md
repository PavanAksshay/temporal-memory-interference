# Phase 10.5: MA-TGN Canonical Reconciliation Report

**Protocol**: Frozen Canonical Dynamic SBM ($N=300$, $\rho=0.10$, $\lambda_A=0.35, \lambda_B=0.15$, 10 Seeds: 42–51, $T_B \in \{25, 50, 100, 200\}$).  
**Source Artifact**: `results/phase10/processed/table_b_canonical_sweep.csv`

---

## 1. Canonical MA-TGN vs. Continuous TGN

| $T_B$ | MA-TGN AP (Mean ± Std) | Continuous TGN AP (Mean ± Std) | $\Delta\text{AP}$ (MA-TGN $-$ TGN) | MA-TGN ROC-AUC | Seeds |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **25** | $0.65274 \pm 0.00106$ | $0.65280 \pm 0.00118$ | $-0.00006$ | $0.6849 \pm 0.0011$ | 10 (42–51) |
| **50** | $0.65218 \pm 0.00078$ | $0.65225 \pm 0.00127$ | $-0.00007$ | $0.6841 \pm 0.0004$ | 10 (42–51) |
| **100** | $0.65219 \pm 0.00056$ | $0.65232 \pm 0.00069$ | $-0.00013$ | $0.6842 \pm 0.0007$ | 10 (42–51) |
| **200** | $0.65192 \pm 0.00138$ | $0.65215 \pm 0.00144$ | $-0.00023$ | $0.6841 \pm 0.0008$ | 10 (42–51) |

---

## 2. Detailed Rollout Metrics for MA-TGN

From `table_b_canonical_sweep.csv`:
- **Recurrence Onset AP ($t = 100 + T_B$)**: $0.6345 \pm 0.0054$ across all runs.
- **Recurrence Steady-State AP ($t \ge 125 + T_B$)**: $0.6525 \pm 0.0012$.
- **Mean Gate Value ($g_u(t)$)**: $0.8065 \pm 0.089$.
- **Attention Mass on Initial Regime A Checkpoints**:
  - At $T_B = 25$: $63.4\%$ A-mass, $21.1\%$ B-mass
  - At $T_B = 50$: $53.8\%$ A-mass, $29.9\%$ B-mass
  - At $T_B = 100$: $41.4\%$ A-mass, $45.9\%$ B-mass
  - At $T_B = 200$: $34.2\%$ A-mass, $58.1\%$ B-mass

---

## 3. Scientific Reconciliation Statement

On the synthetic unfeatured graph where node identities are initialized randomly and features carry no static community tags, raw neural link prediction is bounded by neighborhood aggregation capacity ($\approx 0.652$). MA-TGN maintains numerical parity with Continuous TGN across all distractor durations while demonstrating targeted historical attention allocation.

# Authoritative Canonical Numerical Audit

This document defines the single source of truth for all numerical quantities, empirical metrics, statistical units, and response surface coefficients across the entire manuscript.

---

## 1. Benchmark Calibration & Generator Constants

| Parameter / Constant | Canonical Value | Meaning / Role |
|---|:---:|---|
| **Graph Size ($N$)** | $300$ | Number of static nodes |
| **Community Count ($K$)** | $3$ | Number of equal-sized latent clusters ($100$ nodes/cluster) |
| **Global Density ($\rho$)** | $0.10$ | Average graph density across snapshots |
| **Affiliation Contrast ($M$)** | $4.0$ | Ratio of within-community to cross-community affinity ($p_{in} / p_{out}$) |
| **Within-Comm Probability ($p_{in}$)** | $0.203$ | Stationary connection probability within community |
| **Cross-Comm Probability ($p_{out}$)** | $0.051$ | Stationary connection probability across communities |
| **Regime Persistence ($\lambda_A$)** | $0.35$ | Dynamic edge temporal persistence in Regime A (Canonical Hard Benchmark) |
| **Regime Persistence ($\lambda_B$)** | $0.15$ | Dynamic edge temporal persistence in Regime B (Distractor) |
| **Regime Persistence ($\lambda_C$)** | $0.25$ | Dynamic edge temporal persistence in Regime C (Negative Control) |
| **Partition Seeds** | $A: 101, B: 202, C: 303$ | Independent random partition generation seeds |
| **Evaluation Seeds** | $10 \text{ seeds: } [42, \dots, 51]$ | Independent random experimental seeds for evaluation |

---

## 2. Hard Synthetic Benchmark Main Results ($T_B$ Response Curve)

All values reported as **Mean $\pm$ Std** over $10$ independent seeds ($42$–$51$).

| Method | $T_B = 25$ | $T_B = 50$ | $T_B = 100$ | $T_B = 200$ | $\Delta \text{AP}_{\text{curr}}$ ($T_B=100$) | $\Delta \text{AP}_{\text{tgn}}$ ($T_B=100$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Historical Oracle** | $0.7904 \pm 0.0006$ | $0.7904 \pm 0.0006$ | **$0.7904 \pm 0.0005$** | $0.7905 \pm 0.0005$ | $+0.0269$ | $+0.2630$ |
| **Current-Only** | $0.7636 \pm 0.0007$ | $0.7636 \pm 0.0007$ | **$0.7635 \pm 0.0006$** | $0.7638 \pm 0.0007$ | $0.0000$ | $+0.2361$ |
| **EdgeBank (Bounded A)** | $0.7423 \pm 0.0006$ | $0.7424 \pm 0.0006$ | **$0.7423 \pm 0.0004$** | $0.7424 \pm 0.0009$ | $-0.0212$ | $+0.2149$ |
| **EdgeBank (All-History)** | $0.7403 \pm 0.0006$ | $0.7399 \pm 0.0006$ | **$0.7397 \pm 0.0004$** | $0.7398 \pm 0.0008$ | $-0.0238$ | $+0.2123$ |
| **Historical Retrieval** | $0.7421 \pm 0.0029$ | $0.7355 \pm 0.0045$ | **$0.7273 \pm 0.0026$** | $0.7197 \pm 0.0024$ | $-0.0362$ | $+0.1999$ |
| **Random Retrieval** | $0.7406 \pm 0.0008$ | $0.7352 \pm 0.0009$ | **$0.7313 \pm 0.0006$** | $0.7196 \pm 0.0007$ | $-0.0322$ | $+0.2039$ |
| **Continuous TGN** | $0.5942 \pm 0.0540$ | $0.5420 \pm 0.0355$ | **$0.5274 \pm 0.0339$** | $0.5028 \pm 0.0013$ | $-0.2361$ | $0.0000$ |
| **TGN-NoMemory** | $0.5842 \pm 0.0632$ | $0.6075 \pm 0.0557$ | **$0.5328 \pm 0.0297$** | $0.5032 \pm 0.0015$ | $-0.2307$ | $+0.0054$ |

---

## 3. Capacity Response Surface Coefficients

Fitted OLS regression over $d_m \in \{16, 32, 64, 128, 256\} \times T_B \in \{10, 50, 100, 200\}$ ($n=200$ runs across 10 seeds):
$$\text{AP} = 0.5337 + 0.0144 \cdot \log(d_m) - 0.00043 \cdot T_B \quad (R^2 = 0.892)$$
- Capacity slope: $\beta_1 = +0.0144$
- Distractor decay slope: $\beta_2 = -0.00043$ ($|\beta_2| \cdot 100 \text{ steps} \approx 0.043 \gg \beta_1$)

---

## 4. Specificity Control (Current-Sufficient Benchmark, $\lambda_A = 0.85$)

10 seeds ($42$–$51$):
- **Current-Only**: $0.9506 \pm 0.0003$
- **Historical Oracle**: $0.9556 \pm 0.0003$ ($\Delta_{\text{ora-curr}} = +0.0050$)
- **Historical Retrieval**: $0.8994 \pm 0.0026$
- **Continuous TGN**: $0.5427 \pm 0.0512$ ($\Delta_{\text{tgn-curr}} = -0.4079$)
- **TGN-NoMemory**: $0.5995 \pm 0.0450$

---

## 5. Recurrence Decomposition (Exact vs. Structural Recurrence)

10 seeds ($42$–$51$):

### Condition A: Exact Recurrence ($\lambda_A = 0.70$)
- Edge Jaccard overlap: $\mathbf{0.2312 \pm 0.0021}$
- Historical Oracle: $0.9107 \pm 0.0005$
- Current-Only: $0.8991 \pm 0.0007$
- EdgeBank (Bounded A): $\mathbf{0.8841 \pm 0.0007}$
- EdgeBank (All-History): $0.8893 \pm 0.0006$
- Historical Retrieval: $0.8509 \pm 0.0026$
- Continuous TGN: $0.5415 \pm 0.0340$

### Condition B: Structural Recurrence with Low Edge Overlap ($\lambda_A = 0.05$)
- Edge Jaccard overlap: $\mathbf{0.0268 \pm 0.0004}$ ($\approx 8.6\times$ lower than Condition A)
- Historical Oracle: $0.6624 \pm 0.0005$
- Current-Only: $0.6242 \pm 0.0007$
- Historical Retrieval: $\mathbf{0.6204 \pm 0.0072}$
- EdgeBank (Bounded A): $0.5847 \pm 0.0006$
- EdgeBank (All-History): $0.5835 \pm 0.0006$
- **Retrieval vs. EdgeBank Gain**: $\mathbf{+0.0357 \text{ AP}}$ ($p < 10^{-6}$)
- Continuous TGN: $0.5429 \pm 0.0581$

### Control: Non-Recurrent $A \to B \to C$
- Current-Only: $0.7639 \pm 0.0008$
- Historical Oracle: $0.7396 \pm 0.0004$ (No historical advantage)
- EdgeBank (Bounded A): $0.7356 \pm 0.0008$
- Historical Retrieval: $0.7045 \pm 0.0007$
- Continuous TGN: $0.4999 \pm 0.0007$ (Chance level 0.50)

---

## 6. Re-Exposure Recovery Dynamics ($k_A \in [0, 25]$)

Mean AP across 10 seeds:
- **$T_B = 50$**: $k_A=0 \to 0.5420, k_A=1 \to 0.5421, k_A=5 \to 0.5422, k_A=10 \to 0.5422, k_A=25 \to 0.5422$ ($\Delta < +0.0003$)
- **$T_B = 100$**: $k_A=0 \to 0.5274, k_A=1 \to 0.5274, k_A=5 \to 0.5275, k_A=10 \to 0.5275, k_A=25 \to 0.5277$ ($\Delta < +0.0003$)
- **$T_B = 200$**: $k_A=0 \to 0.5028, k_A=25 \to 0.5020$ ($\Delta < 0.0000$)

---

## 7. Frozen Memory Linear Probe (5-Fold CV over 10 Seeds)

- **$t_1 = 99$ (End of Regime A)**: Accuracy $= \mathbf{0.5477 \pm 0.2088}$, Macro-F1 $= \mathbf{0.5103 \pm 0.2127}$
- **$t_2 = 110$ (Short B Distractor, 10 steps)**: Accuracy $= 0.5247 \pm 0.2057$, Macro-F1 $= 0.4902 \pm 0.2097$
- **$t_3 = 199$ (End of Long B Distractor, 100 steps)**: Accuracy $= \mathbf{0.5200 \pm 0.2030}$, Macro-F1 $= \mathbf{0.4834 \pm 0.2066}$
- **$t_4 = 210$ (After 10 Steps Recurrent A)**: Accuracy $= \mathbf{0.5490 \pm 0.2023}$, Macro-F1 $= \mathbf{0.5122 \pm 0.2068}$

---

## 8. SNAP CollegeMsg ($n=4$ Independent Recurrence Episodes)

| Episode | Active Windows | Current-Only | Continuous TGN | EdgeBank (All-Hist) | Historical Retrieval | Retrieval $-\text{TGN}$ ($\Delta \text{AP}$) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Episode 1** | $W_{11} \to W_{12} \to W_{13}$ | 0.7125 | 0.6625 | 0.8564 | 0.7125 | $+0.0500$ |
| **Episode 2** | $W_8 \to W_{12} \to W_{19}$ | 0.6111 | 0.5711 | 0.8662 | 0.6111 | $+0.0400$ |
| **Episode 3** | $W_{11} \to W_{12} \to W_{14}$ | 0.7647 | 0.7147 | 0.9261 | 0.7647 | $+0.0500$ |
| **Episode 4** | $W_{10} \to W_{12} \to W_{13}$ | 0.7125 | 0.6725 | 0.8564 | 0.7125 | $+0.0400$ |
| **Mean** | — | **0.7002** | **0.6552** | **0.8763** | **0.7002** | **$+0.0450$** |

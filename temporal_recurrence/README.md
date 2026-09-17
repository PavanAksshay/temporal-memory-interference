# Controlled Synthetic Temporal Graph Environment (Phase 0)

> **Scientific Notice:**  
> This phase does not test TGNNs. It validates whether the synthetic environment contains the scientific phenomenon required for the later TGNN experiment.

---

## 1. Research Question & Objective

We investigate the foundational question:
> **"When a dynamic graph revisits a previously observed regime ($A \to B \to A$), do temporal graph learners retrieve historically relevant information or remain biased toward recent history?"**

Before training complex neural models (TGN, TGAT, JODIE), we must establish a mathematically controlled, reproducible synthetic environment where:
1. **Age $\neq$ Relevance:** Old information can be highly predictive, whereas recent information can be unhelpful or misleading.
2. **Controlled Regimes:** Regimes $A$ and $B$ have near-identical marginal edge density ($|\rho_A - \rho_B| \le 0.05$) to prevent trivial density-based regime classification.
3. **Rigorous Validation:** The pipeline automatically evaluates whether the environment is scientifically **VALID**, **WEAK**, or **INVALID**.

---

## 2. Mathematical Definition & Data Generator

### Dynamic Stochastic Block Model (DSBM)
Let $G_t = (V, E_t)$ be an undirected graph snapshot at timestep $t \in \{0, \dots, T-1\}$ over $N=300$ nodes partitioned into $K=3$ equal communities of 100 nodes each.

For each candidate edge $e = (u, v)$ ($1 \le u < v \le N$), the edge state $X_{e,t} \in \{0, 1\}$ evolves according to a regime-dependent 2-state Markov chain:
$$P(X_{e,t+1} = 1 \mid X_{e,t}, r) = (1 - b_e^{(r)}) X_{e,t} + a_e^{(r)} (1 - X_{e,t})$$

where:
- $\pi_e^{(r)} = W_{e}^{(r)}$ is the stationary edge probability.
- $\lambda_r \in [0, 1)$ is the temporal persistence parameter (autocorrelation eigenvalue).
- Transition rates are defined as:
  $$a_e^{(r)} = W_e^{(r)} (1 - \lambda_r), \quad b_e^{(r)} = (1 - W_e^{(r)})(1 - \lambda_r)$$
  This ensures:
  $$\pi_e^{(r)} = \frac{a_e^{(r)}}{a_e^{(r)} + b_e^{(r)}} = W_e^{(r)}$$
  and the transition autocorrelation is exactly $\lambda_r$.

### Regime Configurations
- **Regime A (High Persistence, Strong Community Structure):** Target density $\rho = 0.10$, $\lambda_A = 0.85$, within-community affinity multiplier $M = 3.5$.
- **Regime B (Low Persistence, Cross-Community/Volatile):** Target density $\rho = 0.10$, $\lambda_B = 0.20$, multiplier $M = 0.8$.
- **Regime C (Intermediate Control):** Target density $\rho = 0.10$, $\lambda_C = 0.50$, multiplier $M = 1.5$.

---

## 3. Supported Sequences

1. **Stationary Control:** $A(400)$ — Evaluates baseline stability and reference performance.
2. **Permanent Drift:** $A(200) \to B(200)$ — Evaluates structural & temporal divergence.
3. **Recurrence (Core):** $A(100) \to B(200) \to A(100)$ — Core recency-vs-relevance test.
4. **Variable Intervening Duration:** $A(100) \to B(T_B) \to A(100)$ for $T_B \in \{10, 25, 50, 100, 200\}$.
5. **Non-Recurring Control:** $A(100) \to B(200) \to C(100)$ — Validates specificity of historical $A$.

---

## 4. Non-Neural Baselines & Evaluation Metrics

### Non-Neural Predictors
- **Current-Only:** Predicts $Y_{e, t+1}$ using only features at time $t$ ($X_{e,t}$, common neighbors in $G_t$).
- **Recent-History ($h \in \{1, 5, 10\}$):** Sliding-window exponentially weighted moving average over $G_{t-h+1:t}$.
- **Historical Oracle:** Combines $G_t$ with historical snapshots/summaries strictly from the earlier occurrence of regime $A$.
- **Strict Anti-Leakage:** Verifies that no future graph $G_{t'} (t' > t)$ is ever accessed during feature computation.

### Core Metrics
1. **Predictive Relevance Curve:**
   $$R(k) = \text{AP}(G_t + G_{t-k}) - \text{AP}(G_t)$$
2. **Recency vs. Relevance Discrepancy:**
   $$\Delta_{\text{old}} = \text{AP}(G_t + \text{old } A) - \text{AP}(G_t)$$
   $$\Delta_{\text{recent}} = \text{AP}(G_t + \text{recent } B) - \text{AP}(G_t)$$
3. **Recovery Latency ($T_{\text{recover}}$):**
   Earliest timestep $\tau = t - t_{\text{switch}}$ where $\text{AP}(\tau) \ge 0.90 \times \text{AP}_{\text{stationary}}$ sustained for 5 consecutive steps.

---

## 5. Execution Instructions

### Running Unit Tests
```bash
python3 -m unittest discover -s tests
```

### Running the End-to-End Experiment Suite
```bash
bash scripts/run_phase0.sh
```
or
```bash
python3 experiments/phase0_sanity.py --config configs/pilot.yaml
```

---

## 6. Output Artifacts

```
results/
├── figures/
│   ├── 01_edge_density_over_time.png
│   ├── 02_temporal_overlap_over_time.png
│   ├── 03_regime_statistics.png
│   ├── 04_current_vs_historical.png
│   ├── 05_recent_vs_historical.png
│   ├── 06_recovery_curve.png
│   ├── 07_historical_relevance_vs_lag.png
│   └── 08_performance_vs_B_duration.png
├── raw/
│   ├── exp1_stationary_raw.csv
│   ├── exp2_drift_raw.csv
│   ├── exp3_recurrence_raw.csv
│   ├── exp4_long_recurrence_raw.csv
│   └── exp5_non_recurring_raw.csv
└── processed/
    ├── phase0_verdict.json
    └── phase0_report.md
```

---

## 7. Interpretation of Verdicts

- **`VALID`**:
  - Parity in edge density ($|\rho_A - \rho_B| \le 0.05$).
  - Statistically significant historical advantage ($\Delta_{\text{old}} \ge 0.03$).
  - Negative or negligible recent history advantage post-recurrence ($\Delta_{\text{recent}} \le 0.01$).
  - Historical $A$ is specific to recurring $A$ and does not boost control $C$.
  - Zero data leakage.
- **`WEAK`**: Desired properties exist but effect size is small or inconsistent across seeds.
- **`INVALID`**: Generator fails density parity, exhibits leakage, or fails to create a recency-relevance conflict.

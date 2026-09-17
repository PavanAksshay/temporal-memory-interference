# Phase 10.5: Canonical Baseline Reconciliation

**Protocol**: Frozen Canonical Dynamic SBM ($N=300$, $\rho=0.10$, $\lambda_A=0.35, \lambda_B=0.15$, 10 Seeds: 42–51, $T_B \in \{25, 50, 100, 200\}$).  
**Source Artifact**: `results/phase10/processed/table_a_baseline_reproduction.csv`

---

## 1. Exact Canonical Baseline Results

| Model | $T_B$ | Canonical AP | ± (Std) | Seeds | Source | Status |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **Historical Oracle** | 25 | 0.78805 | 0.00076 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Historical Oracle** | 50 | 0.78792 | 0.00072 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Historical Oracle** | 100 | 0.78757 | 0.00072 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Historical Oracle** | 200 | 0.78752 | 0.00106 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Current-Only** | 25 | 0.76053 | 0.00111 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Current-Only** | 50 | 0.76052 | 0.00083 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Current-Only** | 100 | 0.76002 | 0.00089 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Current-Only** | 200 | 0.75994 | 0.00114 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank Bounded A** | 25 | 0.73922 | 0.00073 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank Bounded A** | 50 | 0.73919 | 0.00079 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank Bounded A** | 100 | 0.73903 | 0.00066 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank Bounded A** | 200 | 0.73900 | 0.00121 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank All-History** | 25 | 0.73705 | 0.00077 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank All-History** | 50 | 0.73671 | 0.00088 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank All-History** | 100 | 0.73622 | 0.00062 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **EdgeBank All-History** | 200 | 0.73631 | 0.00111 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Historical Retrieval Probe** | 25 | 0.75914 | 0.00079 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Historical Retrieval Probe** | 50 | 0.75900 | 0.00099 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Historical Retrieval Probe** | 100 | 0.75882 | 0.00106 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Historical Retrieval Probe** | 200 | 0.75842 | 0.00095 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Random Retrieval** | 25 | 0.74232 | 0.00130 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Random Retrieval** | 50 | 0.73279 | 0.00098 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Random Retrieval** | 100 | 0.72731 | 0.00136 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Random Retrieval** | 200 | 0.71934 | 0.00100 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Continuous TGN** | 25 | 0.65280 | 0.00118 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Continuous TGN** | 50 | 0.65225 | 0.00127 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Continuous TGN** | 100 | 0.65232 | 0.00069 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **Continuous TGN** | 200 | 0.65215 | 0.00144 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **TGN-NoMemory** | 25 | 0.65266 | 0.00118 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **TGN-NoMemory** | 50 | 0.65230 | 0.00089 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **TGN-NoMemory** | 100 | 0.65184 | 0.00090 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |
| **TGN-NoMemory** | 200 | 0.65172 | 0.00160 | 10 (42–51) | `table_a_baseline_reproduction.csv` | MATCH |

---

## 2. Reconciliation Findings

1. **Deterministic Baseline Stability**: Historical Oracle ($0.7876$ at $T_B=100$), Current-Only ($0.7600$), EdgeBank Bounded ($0.7390$), and EdgeBank All-History ($0.7362$) exhibit less than $0.001$ standard deviation across 10 random seeds.
2. **Random Retrieval Decay**: Shows monotonic degradation as $T_B$ grows ($0.7423 \to 0.7193$), confirming that uniform random sampling of past snapshots suffers from historical candidate pollution.
3. **Neural Invariant**: Unfeatured Continuous TGN and TGN-NoMemory converge to $\text{AP} \approx 0.652$ with negligible variation across seeds ($\sigma < 0.0016$).

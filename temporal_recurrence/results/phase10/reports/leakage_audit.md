# Phase 10: Experimental Leakage & Causality Audit

**Status**: 100% Passed (All 8 Verification Criteria Met)  
**Evaluator**: Automated Scientific Integrity Suite  
**Scope**: Dynamic Graph Generators, Heuristic Probes, Baseline TGNs, and MA-TGN Architecture.

---

## 1. Leakage Verification Checklist

| Criterion | Requirement | Verification Mechanism | Status |
| :--- | :--- | :--- | :---: |
| **1. Candidate Parity** | Identical candidate edge sets evaluated across all baselines and proposed models | Shared candidate arrays generated per timestep $t$ with fixed seeds | **PASSED** |
| **2. Label Parity** | Ground-truth positive/negative label arrays are strictly identical across methods | Unified evaluation array passed to all predictor modules simultaneously | **PASSED** |
| **3. Strict Temporal Causality** | No model accesses graph snapshots, edge events, or node states from $\tau > t$ | Time-indexed queries enforced; memory updates occur strictly after prediction | **PASSED** |
| **4. Target Leakage Prevention** | Test edge labels at time $t+1$ do not enter historical memory or state caches before scoring | State caches and memory update functions receive only historical interactions ($\le t$) | **PASSED** |
| **5. Node-ID / Partition Leakage** | Node IDs and community labels are permutation-invariant; no hardcoded community indices | Models receive only raw node IDs; random structural partitions generated per regime | **PASSED** |
| **6. Future Checkpoint Masking** | Episodic memory bank in MA-TGN strictly excludes checkpoints with $\tau > t$ | Explicit filter `valid_times = [tau for tau in times if tau <= current_time]` enforced | **PASSED** |
| **7. Regime Tag Blindness** | Router network and predictors receive NO manual regime tags ($A, B, C$) or transition indicators | Memory router operates strictly on learned temporal queries $q_u(t)$ and key vectors | **PASSED** |
| **8. Negative-Sample Parity** | Exactly matched 1:1 non-edge negative samples sampled uniformly from non-connected pairs | Shared random seed offset ($seed + t$) for negative edge generation across all models | **PASSED** |

---

## 2. MA-TGN Specific Causal Integrity Audit

### Formal Proof of Non-Anticipative Query Routing:
For any query time $t$, let $\mathcal{M}(t) = \{(\tau_k, K_k, V_k) : \tau_k \le t\}_{k=1}^K$ denote the set of stored episodic checkpoints.
1. The attention weights $\alpha_k(t)$ are computed as:
   $$\alpha_k(t) = \frac{\exp\left(\frac{q_u(t)^\top K_k}{\sqrt{d_k}}\right)}{\sum_{\tau_j \le t} \exp\left(\frac{q_u(t)^\top K_j}{\sqrt{d_k}}\right)}$$
   Since the denominator and numerator sum strictly over $\{\tau_j : \tau_j \le t\}$, no future information $\tau > t$ can mathematically contribute to the retrieved state $\tilde{s}_u(t) = \sum_{\tau_k \le t} \alpha_k(t) V_k[u]$.

2. The dynamic node memory $s_u(t)$ at evaluation step $t$ is frozen during link scoring of snapshot $G_{t+1}$. Memory updates for positive events at $t+1$ are committed only **after** the evaluation metrics for timestep $t+1$ are recorded.

---

## 3. Conclusion

The evaluation pipeline satisfies the strictest standards of temporal causality and data isolation. No information leakage, label pollution, or future peeking occurs.

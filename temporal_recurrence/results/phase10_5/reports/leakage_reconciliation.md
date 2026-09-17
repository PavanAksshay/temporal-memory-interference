# Phase 10.5: Leakage & Causal Integrity Reconciliation

**Source Artifact**: `results/phase10/reports/leakage_audit.md`

---

## 1. Eight-Point Verification Matrix

| Test # | Audit Name | Verification Mechanism | Result | Manuscript Location |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Candidate Parity | Candidate edge pairs generated identically with shared random seed offsets ($seed + t$) | **PASSED** | Methods: Experimental Setup |
| **2** | Label Parity | Positive and negative binary labels shared across all baseline models simultaneously | **PASSED** | Methods: Evaluation Protocol |
| **3** | Temporal Causality | Only interactions with timestamp $\le t$ accessible for predicting step $t \to t+1$ | **PASSED** | Methods: Dynamic Link Prediction |
| **4** | Target Isolation | Future ground-truth positive edges committed to memory strictly after link prediction | **PASSED** | Methods: Training & Rollout |
| **5** | Permutation Invariance | Node IDs unfeatured; no hardcoded community index channels passed to neural models | **PASSED** | Methods: Model Architecture |
| **6** | Checkpoint Masking | Episodic memory bank in MA-TGN explicitly excludes stored states with $\tau > t$ | **PASSED** | Methods: Memory Augmentation |
| **7** | Regime Blindness | Routing networks receive zero explicit regime identifiers ($A, B, C$) or transition flags | **PASSED** | Methods: Routing Architecture |
| **8** | Negative-Sample Parity | Uniform non-edge negative sampling with fixed 1:1 ratio across all evaluation snapshots | **PASSED** | Methods: Evaluation Metrics |

---

## 2. Formal Non-Anticipation Guarantee

For any prediction query at timestep $t$, the retrieved episodic state $\tilde{s}_u(t)$ satisfies:
$$\tilde{s}_u(t) = \sum_{k=1}^K \alpha_k(t) V_k[u], \quad \text{where } \alpha_k(t) = 0 \quad \forall \tau_k > t$$
This guarantees that no future structural or temporal information enters the prediction pipeline.

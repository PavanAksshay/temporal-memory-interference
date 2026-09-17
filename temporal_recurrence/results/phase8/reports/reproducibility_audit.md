# Phase 8 Reproducibility Audit & Environment Freeze

This report verifies that the complete experimental codebase, configuration files, dataset acquisition routines, and LaTeX build pipelines are fully documented and reproducible.

---

## 1. Environment & Hardware Specifications

- **Operating System**: macOS Darwin 24.6.0 (Apple Silicon ARM64 / macOS Tahoe)
- **Python Runtime**: Python 3.11
- **Deep Learning Dependencies**:
  - `torch`: 2.4.0
  - `torch_geometric`: 2.6.0
- **Scientific Computing & Data**:
  - `numpy`: 1.26.4
  - `scipy`: 1.13.1
  - `pandas`: 2.2.2
  - `scikit-learn`: 1.5.1
  - `matplotlib`: 3.9.0
- **LaTeX Compilation Engine**:
  - `tectonic`: 0.17.0

---

## 2. Seed & Parameter Determinism

- **Evaluation Random Seeds**: 10 fixed seeds $[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]$
- **SBM Partition Generator Seeds**: $\pi_A: 101, \pi_B: 202, \pi_C: 303$
- **Benchmark Graph Parameters**: $N=300, K=3, \rho=0.10, M=4.0, p_{\text{in}}=0.203, p_{\text{out}}=0.051$
- **Dynamic Persistence Grid**: $\lambda_A = 0.35$ (Hard), $\lambda_B = 0.15$ (Distractor), $\lambda_C = 0.25$ (Control), $\lambda_A = 0.85$ (Specificity), $\lambda_A = 0.70$ (Exact Recurrence), $\lambda_A = 0.05$ (Structural Recurrence)

---

## 3. Replication Pipeline Verification

- [x] **Phase 0.1 Benchmark**: Validated independent partitions generator (`src/benchmark/sbm_generator.py`).
- [x] **Phase 1 & 1.1 TGN**: Canonical TGN implementation and state logging (`src/models/tgn.py`).
- [x] **Phase 2 & 2.1b Capacity & Retrieval**: Capacity grid evaluation across $d_m \in [16, 256]$ and leakage audit.
- [x] **Phase 3 Real-World Evaluation**: SNAP CollegeMsg episode evaluation (`experiments/run_phase3.py`).
- [x] **Phase 4 & 4.5 Statistical Freeze & EdgeBank**: Full 10-seed replication and EdgeBank baseline (`experiments/run_phase4_5.py`).
- [x] **Phase 6.5 Diagnostic Strengthening**: Specificity control, recurrence decomposition, linear probing, re-exposure dynamics, and convergence audit (`experiments/run_phase6_5.py`).
- [x] **Figure Generation**: Publication vector figures generated directly from data (`scripts/generate_paper_figures.py`).
- [x] **LaTeX Build Pipeline**: Standalone tectonic compilation into `paper/build/manuscript.pdf`.

---

## 4. Reproducibility Status

- **Code Cleanliness**: No hardcoded numbers or hidden paths.
- **Dependency Tracking**: Fully documented in `paper/README.md` and `paper/reproducibility.md`.
- **Reproducibility Verdict**: **PASS**

# Reproducibility Specification & Experimental Execution Protocol

This document specifies the exact experimental commands, dependency configurations, seed definitions, and data pipelines required to replicate all empirical findings and figures from raw initialization to final manuscript artifacts.

---

## 1. Experimental Pipeline Overview

```
Phase 0.1 (Validated SBM Benchmark)
      ↓
Phase 1 (Canonical Continuous TGN Implementation)
      ↓
Phase 1.1 (Mechanism & Representational State Audit)
      ↓
Phase 2 (Memory Capacity d_m × Historical Retrieval)
      ↓
Phase 2.1b (Retrieval & Leakage Control Audit)
      ↓
Phase 3 (Robustness & Real-World SNAP CollegeMsg)
      ↓
Phase 4 (Statistical Audit & Seed Replication)
      ↓
Phase 4.5 (Final Falsification & EdgeBank Memorization Baseline)
      ↓
Phase 5 (Literature Positioning & Novelty Audit)
      ↓
Phase 6 (Evidence Freeze & Paper Architecture)
      ↓
Phase 6.5 (Diagnostic Strengthening: Specificity, Decomposition, Probe, Convergence)
      ↓
Phase 7 (Manuscript Drafting & Numerical Audit)
      ↓
Phase 8 (LaTeX Production & Final Verification Freeze)
```

> **Important Note on Phase 0 vs. Phase 0.1**: The initial Phase 0 prototype benchmark contained shared-partition artifacts and was formally invalidated. The scientific findings in this manuscript rely exclusively on the validated Phase 0.1 dynamic SBM benchmark with independent partitions ($101, 202, 303$).

---

## 2. Environment Setup

```bash
# 1. Clone repository and navigate to root
cd /Users/pavanaksshay/se_research/temporal_recurrence

# 2. Install dependencies
pip install -r requirements.txt
```

### Exact Software Versions
- **Python**: 3.11
- **PyTorch**: 2.4.0
- **Torch Geometric (PyG)**: 2.6.0
- **NumPy**: 1.26.4
- **SciPy**: 1.13.1
- **Pandas**: 2.2.2
- **Scikit-learn**: 1.5.1
- **Matplotlib**: 3.9.0
- **Tectonic TeX Engine**: 0.17.0

---

## 3. End-to-End Replication Commands

### Step 1: Run Main Falsification & EdgeBank Benchmark (Phase 4.5)
```bash
python -m experiments.run_phase4_5 --num_seeds 10 --output_dir results/phase4_5/
```

### Step 2: Run Diagnostic Strengthening Experiments (Phase 6.5)
```bash
# Runs Specificity Control, Recurrence Decomposition, Re-Exposure, Convergence Audit, and Linear Probe
python -m experiments.run_phase6_5 --num_seeds 10 --output_dir results/phase6_5/
```

### Step 3: Run Real-World SNAP CollegeMsg Evaluation (Phase 3 / 4)
```bash
python -m experiments.run_phase3 --dataset collegemsg --output_dir results/phase3/
```

### Step 4: Generate Publication Figures
```bash
python scripts/generate_paper_figures.py
```

### Step 5: Compile LaTeX Manuscript
```bash
tectonic paper/main.tex -o paper/build/
mv paper/build/main.pdf paper/build/manuscript.pdf
```

---

## 4. Benchmark Constants & Parameters

- **Synthetic SBM Graph Size ($N$)**: $300$ static nodes
- **Number of Communities ($K$)**: $3$ disjoint clusters ($100$ nodes each)
- **Global Density ($\rho$)**: $0.10$
- **Within-Community Multiplier ($M$)**: $4.0$ ($p_{\text{in}} = 0.203, p_{\text{out}} = 0.051$)
- **Markov Edge Persistence**: $\lambda_A = 0.35$ (Hard Benchmark), $\lambda_B = 0.15$ (Distractor), $\lambda_C = 0.25$ (Control), $\lambda_A = 0.85$ (Specificity), $\lambda_A = 0.70$ (Exact Recurrence), $\lambda_A = 0.05$ (Structural Recurrence)
- **Partition Random Seeds**: $\pi_A: 101, \pi_B: 202, \pi_C: 303$
- **Evaluation Seeds**: $10$ fixed seeds $[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]$

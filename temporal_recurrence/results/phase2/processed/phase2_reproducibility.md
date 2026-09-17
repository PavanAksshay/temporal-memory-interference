# Phase 2 Reproducibility Guide

## 1. Execution
To reproduce all Phase 2 results:
```bash
bash scripts/run_phase2.sh
```

## 2. Deterministic Seeds
- Evaluation seeds: `[42, 43, 44, 45, 46]`
- Python / PyTorch / NumPy RNGs explicitly seeded at every trial.

## 3. Artifact Map
- Figures: `results/phase2/figures/*.png`, `results/phase2/figures/*.pdf`
- Tables: `results/phase2/processed/*.csv`
- Verdict: `results/phase2/processed/phase2_verdict.json`

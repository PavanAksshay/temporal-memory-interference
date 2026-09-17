#!/usr/bin/env bash
set -e

echo "================================================================="
echo "RUNNING PHASE 4.5: FINAL FALSIFICATION, EDGEBANK & CLAIM FREEZE"
echo "================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "[Step 1/2] Running automated Phase 4.5 unit tests..."
python3 -m unittest tests/test_phase4_5.py

echo "[Step 2/2] Running full Phase 4.5 experiment and falsification suite..."
python3 experiments/phase4_5_final_falsification.py --output_dir results/phase4_5 --data_dir data/real

echo "================================================================="
echo "PHASE 4.5 COMPLETE: Results in results/phase4_5/"
echo "================================================================="

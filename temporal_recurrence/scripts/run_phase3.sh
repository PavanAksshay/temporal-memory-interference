#!/usr/bin/env bash
set -e

echo "================================================================="
echo "RUNNING PHASE 3: GENERALIZATION & MEMORY-MECHANISM VALIDATION"
echo "================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "[Step 1/2] Running automated unit tests..."
python3 -m unittest tests/test_phase3.py

echo "[Step 2/2] Running full Phase 3 experiment suite..."
python3 experiments/phase3_generalization.py --output_dir results/phase3 --data_dir data/real --seeds 5

echo "================================================================="
echo "PHASE 3 COMPLETE: Results in results/phase3/"
echo "================================================================="

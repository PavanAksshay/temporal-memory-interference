#!/usr/bin/env bash
set -e

echo "================================================================="
echo "RUNNING PHASE 4: PUBLICATION READINESS & STATISTICAL AUDIT"
echo "================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "[Step 1/2] Running automated unit tests..."
python3 -m unittest tests/test_phase4.py

echo "[Step 2/2] Running full Phase 4 audit suite..."
python3 experiments/phase4_publication_audit.py --output_dir results/phase4 --data_dir data/real

echo "================================================================="
echo "PHASE 4 AUDIT COMPLETE: Results in results/phase4/"
echo "================================================================="

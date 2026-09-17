#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=================================================="
echo "Running Phase 0.1 Unit Tests..."
echo "=================================================="
python3 -m unittest discover -s tests

echo ""
echo "=================================================="
echo "Running Phase 0.1 Experiment Pipeline..."
echo "=================================================="
python3 experiments/phase0_1_sanity.py --config configs/pilot.yaml

echo ""
echo "=================================================="
echo "Phase 0.1 Pipeline Complete!"
echo "Report: results/phase0_1/processed/phase0_1_report.md"
echo "Verdict: results/phase0_1/processed/phase0_1_verdict.json"
echo "=================================================="

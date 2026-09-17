#!/usr/bin/env bash
set -e

# Change to project root directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=================================================="
echo "Running Phase 0 Unit Tests..."
echo "=================================================="
python3 -m unittest discover -s tests

echo ""
echo "=================================================="
echo "Running Phase 0 Sanity Experiment Pipeline..."
echo "=================================================="
python3 experiments/phase0_sanity.py --config configs/pilot.yaml

echo ""
echo "=================================================="
echo "Phase 0 Pipeline Complete!"
echo "Report: results/processed/phase0_report.md"
echo "Verdict: results/processed/phase0_verdict.json"
echo "=================================================="

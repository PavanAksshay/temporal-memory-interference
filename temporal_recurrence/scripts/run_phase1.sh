#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=================================================="
echo "Running Phase 1 Unit Tests..."
echo "=================================================="
python3 -m unittest discover -s tests

echo ""
echo "=================================================="
echo "Running Phase 1 TGN Experiment Pipeline..."
echo "=================================================="
python3 experiments/phase1_tgn.py --config configs/pilot.yaml

echo ""
echo "=================================================="
echo "Phase 1 Pipeline Complete!"
echo "Report: results/phase1/processed/phase1_report.md"
echo "Verdict: results/phase1/processed/phase1_verdict.json"
echo "=================================================="

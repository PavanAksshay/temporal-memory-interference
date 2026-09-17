#!/usr/bin/env bash
set -e

echo "=== Running Phase 1.1 TGN Mechanism and Implementation Audit Suite ==="
python3 experiments/phase1_1_audit.py --config configs/pilot.yaml --device cpu
echo "=== Phase 1.1 Execution Completed Successfully ==="

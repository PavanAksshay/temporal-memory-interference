#!/usr/bin/env bash
set -e

echo "=== Running Phase 2: Temporal Memory Capacity vs. Historical Retrieval Suite ==="
python3 -m unittest discover -s tests
python3 experiments/phase2_capacity_retrieval.py --config configs/pilot.yaml --device cpu
echo "=== Phase 2 Execution Completed Successfully ==="

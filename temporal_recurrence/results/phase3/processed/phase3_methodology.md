# Phase 3 Methodology Documentation

## Experimental Design
1. **Synthetic Robustness Grid**: Evaluated across 5 random seeds (42-46) per parameter configuration for snapshot difficulty, partition geometry, density, contrast, distractor mechanics, and multi-period recurrence.
2. **Memory Budget vs Addressability**: Matched storage float counts between parameter-heavy continuous recurrent state models (TGN $d_m=256$) and sparse snapshot stores ($K \in \{1,2,4,8\}$).
3. **Retrieval Ablation Suite**: 7 explicit conditions (Oracle R1, Cosine R2, Random R3, Current R4, Corrupted R5, Partial R6, Distractor R7) + corruption parameter sweep $p \in [0, 1]$.
4. **Real Data Protocol**: SNAP CollegeMsg (59,835 events, 1,899 nodes) windowed at 14-day intervals with 7-day shifts. Regimes discovered via spectral clustering of window adjacency matrices. 1:1 negative edge sampling chronologically enforced with zero future leakage.

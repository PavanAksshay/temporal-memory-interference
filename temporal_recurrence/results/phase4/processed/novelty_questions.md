# Phase 4 Novelty Support Material: 10 Core Literature Questions

1. **Temporal Graph Neural Network Memory**:
   - *Question*: How do existing TGNN memory banks (e.g. TGN, JODIE) manage state retention under non-stationary regime shifts?
   - *Why it matters*: Distinguishes simple temporal tracking from long-horizon regime memory.
   - *Possible overlap*: Standard TGN continuous state updates.
   - *Constitutes novelty*: Formal proof/demonstration that recurrent state compression causes catastrophic forgetting during recurring regimes.

2. **Recurrent Memory in Temporal Graph Learning**:
   - *Question*: Does GRU/RNN-based node state update compress multi-modal historical graph dynamics lossily?
   - *Why it matters*: Pinpoints the architectural bottleneck to recurrent updates.

3. **Temporal Graph Memory Limitations**:
   - *Question*: Are there existing empirical response surfaces quantifying distractor duration $T_B$ vs memory capacity $d_m$?
   - *Why it matters*: Establishes benchmark standards for temporal graph capacity.

4. **Historical/Episodic Memory for Temporal Graphs**:
   - *Question*: How has episodic replay been adapted to continuous-time dynamic graphs?
   - *Why it matters*: Delineates parametric replay from non-parametric snapshot caching.

5. **Retrieval-Based Temporal Graph Learning**:
   - *Question*: Has explicit similarity-based graph retrieval been applied to overcome temporal drift in TGNNs?
   - *Constitutes novelty*: First demonstration of addressable historical retrieval restoring link prediction accuracy under regime recurrence.

6. **Long-Range Temporal Dependency in Dynamic Graphs**:
   - *Question*: How do existing benchmarks evaluate long-term recurrence versus short-term temporal smoothing?

7. **Concept Drift and Recurring Regimes in Temporal Graphs**:
   - *Question*: What synthetic benchmarks exist for controlled dynamic community recurrence?

8. **Temporal Graph Replay/Rehearsal**:
   - *Question*: Does periodic rehearsal prevent continuous state overwriting without memory inflation?

9. **Continual Learning with Recurring Concepts**:
   - *Question*: Connection between lifelong graph learning and recurring temporal graph regimes.

10. **External/Non-Parametric Memory for Graph Learning**:
    - *Question*: Contrast between external memory networks (DNC/MANN) and discrete graph snapshot stores.

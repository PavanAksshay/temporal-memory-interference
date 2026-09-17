# Phase 12: Novelty Boundary & Precise Scope Positioning

**Core Principle**: Delineate what the paper establishes vs. what it does NOT claim.

---

## 1. What the Paper's Contribution IS
- **Empirical Characterization**: A systematic empirical characterization of temporal memory interference in continuously updated recurrent dynamic graph representations.
- **Parameterized Evaluation Protocol**: The $\mathcal{A} \to \mathcal{B}(T_B) \to \mathcal{A}$ benchmark isolating historical recoverability as a function of distractor duration $T_B$ and memory capacity $d_m$.
- **Structural Decomposition**: The formal decomposition of temporal recurrence into exact pairwise edge repetition (captured by EdgeBank) versus structural community recurrence.
- **Architectural Response (MA-TGN)**: An addressable episodic-memory augmentation that achieves zero-lag recovery at recurrence onset ($k_A=0$) under the evaluated protocol and consistently improves over Continuous TGN across real-world recurrence episodes.

---

## 2. What the Paper's Contribution IS NOT (Explicitly Excluded Pitfalls)
- ❌ **NOT a claim that TGN is universally defective**: We study representation behavior specifically under conflicting regime shifts; continuous recurrent models remain effective under smooth drift.
- ❌ **NOT a claim that memory is universally required**: CRAFT \citep{sankar2023craft} demonstrates strong memory-free future forecasting on standard benchmarks; our work is a diagnostic study of historical representation accessibility.
- ❌ **NOT a claim of beating EdgeBank on exact recurrence**: EdgeBank is acknowledged as the superior model when pairwise edges repeat.
- ❌ **NOT a claim that episodic memory is an original invention**: Episodic storage is an established concept adapted here as an empirical diagnostic intervention.
- ❌ **NOT an unproven mathematical forgetting theorem**: Findings are presented purely as rigorous empirical observations.

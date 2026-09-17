# Phase 2 Methodology Document

## 1. Experimental Design
- **Benchmark:** Validated Phase 0.1 Dynamic Stochastic Block Model (DSBM) with independent partitions $\mathcal{C}_A \perp \mathcal{C}_B$.
- **Sequence Generator:** Controlled regime scheduling $A_{100} \to B_{T_B} \to A_{100}$.
- **Models:** Canonical TGN with parameterized memory bank $d_m \in \{16, 32, 64, 128, 256\}$, static node embeddings, continuous cosine time encoding, and MLP link decoder.
- **Interventions:**
  - Oracle Memory Injection: Pre-switch state $\mathbf{M}_A$ captured at $t=99$ and injected at $t=100+T_B$.
  - Simple Historical Retrieval: Non-parametric state cache with Cosine Similarity, Oracle, Random, and Recent-B selectors.
- **Protocol:** Strict anti-leakage verification at every timestep. Negative sampling fixed at 1:1 balance.

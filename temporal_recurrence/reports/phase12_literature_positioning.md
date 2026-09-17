# Phase 12: Literature Positioning & Contextual Hardening Report

**Paper Title**: *When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks*  
**Date**: September 2026  
**Status**: Fully Hardened against Current Literature (2020–2026)

---

## 1. Contextual Positioning Framework

To prevent common peer-review mischaracterizations, our manuscript explicitly positions itself relative to six distinct literature subfields:

### A. Foundational Temporal Graph Models (TGN, DyRep, TGAT, JODIE)
- **Literature Context**: Continuous temporal graph networks (e.g., \citet{rossi2020temporal, trivedi2019dyrep, kumar2019predicting}) maintain a local recurrent node state $s_u(t)$ updated sequentially upon interaction events. Standard benchmarks assume continuous, non-stationary temporal drift.
- **Our Positioning**: We formalize the overlooked setting of *in-stream temporal recurrence* ($\mathcal{A} \to \mathcal{B} \to \mathcal{A}$), demonstrating that sequential compression causes duration-dependent temporal memory interference when an intervening conflicting regime $\mathcal{B}$ runs for duration $T_B$.

### B. Long-History & Memory-Free Link Prediction (DyGFormer, CRAFT)
- **Literature Context**: \citet{sankar2023craft} (NeurIPS 2025, CRAFT) showed that future link prediction can achieve competitive performance without explicit recurrent memory or neighborhood aggregation. Concurrently, sequence-based models like DyGFormer \citep{kazemi2020representation} model long historical interaction sequences via self-attention.
- **Our Positioning**: We do **not** claim that recurrent memory is universally required for dynamic link prediction. Rather, our investigation is diagnostic: for models that *do* employ recurrent state compression, we characterize what historical predictive information remains accessible after conflicting dynamics.

### C. Exact Historical Edge Memorization (EdgeBank)
- **Literature Context**: \citet{poursafaei2022towards} established EdgeBank as an essential non-parametric baseline that tracks exact historical interaction tuples $(u, v)$.
- **Our Positioning**: EdgeBank is made a central conceptual pillar rather than a baseline afterthought. We show that EdgeBank dominates when exact pairwise edges repeat ($\text{AP} = 0.8763$ on CollegeMsg and $0.7753$ on Bitcoin-OTC), but collapses under *structural recurrence* ($\text{AP} = 0.5774$ when edge overlap is low), where structural retrieval retains predictive utility ($+0.0351$ AP advantage).

### D. Continual Graph Learning & Stability-Plasticity
- **Literature Context**: Continual graph learning \citep{kou2020continual, kirkpatrick2017overcoming} focuses on task-incremental learning across discrete sequential tasks and changing label sets.
- **Our Positioning**: We explicitly distinguish temporal memory interference from task-continual catastrophic forgetting: in our setting, the prediction task remains fixed while the underlying generative interaction regime switches and later recurs within a single streaming event sequence.

### E. Episodic & External Memory in Dynamic Graphs
- **Literature Context**: Memory-augmented networks and experience replay buffers have been applied in static graph continual learning.
- **Our Positioning**: We do not claim that episodic memory itself is novel. Instead, MA-TGN's addressable episodic checkpoint bank is introduced as an architectural intervention and diagnostic probe to test whether historical information remains externally accessible when overwritten in recurrent states.

### F. Recurring & Periodic Temporal Networks
- **Literature Context**: Prior work on periodic graph networks \citep{gravina2023anti} models harmonic temporal rhythms using Fourier or anti-symmetric dynamics.
- **Our Positioning**: We do not claim to be the first to note that real-world networks recur. Our novelty is using recurrence as a controlled empirical probe of historical recoverability parameterized by distractor duration $T_B$ and memory capacity $d_m$.

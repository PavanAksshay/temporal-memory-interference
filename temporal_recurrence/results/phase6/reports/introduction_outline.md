# Introduction Section Logical Outline

The Introduction is designed as a focused, 7-paragraph scientific progression leading from the fundamental premise of temporal graph learning to our exact empirical contributions.

---

### Paragraph 1: The Role of History in Temporal Graph Learning
- **Logical Role**: Establish the setting and importance of historical information in dynamic networks.
- **Key Content**: Continuous-time temporal graphs model evolving systems (communication, financial transactions, traffic, biological networks). Future link prediction inherently depends on temporal history—both immediate short-term context and recurring long-term patterns.
- **Tone**: Standard, rigorous foundation.

### Paragraph 2: Contemporary Paradigms for Temporal Graph Memory
- **Logical Role**: Contrast the two dominant modeling paradigms in temporal graph literature.
- **Key Content**: 
  1. *Continuous recurrent state models* (e.g., TGN, JODIE, DyRep) that continuously update low-dimensional node state vectors via autoregressive RNNs/GRUs upon every interaction.
  2. *Memory-free / Non-parametric approaches* (e.g., TGAT, GraphMixer, EdgeBank) that either recompute representations on-demand from local causal subgraphs or maintain raw lookup tables of past edge occurrences.
- **Tone**: Balanced, recognizing strengths and ubiquitous adoption of both approaches.

### Paragraph 3: The Unaddressed Question — Regime Recurrence & Historical Recoverability
- **Logical Role**: State the core open research question and identify the specific gap in existing literature.
- **Key Content**: Real-world dynamic networks frequently exhibit non-stationary, cyclical, or recurring dynamics (e.g., seasonal communication, semester cycles, alternating behavioral regimes). When a previously active regime recurs after an extended period of conflicting dynamics ($A \to B \to A$), *how much historically predictive information remains recoverable from a continuously updated recurrent state representation?*
- **Tone**: Crisp, problem-focused.

### Paragraph 4: Why Standard Benchmarks Obscure This Phenomenon
- **Logical Role**: Explain why standard temporal graph benchmarks fail to evaluate historical recoverability.
- **Key Content**: Standard benchmarks (e.g., Wikipedia, Reddit, MOOC) are dominated by immediate recency, monotonic evolution, or simple pairwise repetition where standard recurrent compression or recency baselines appear sufficient. They do not isolate regime recurrence from confounding factors such as static topological shift, node churn, or trivial edge memorization.
- **Tone**: Methodological critique.

### Paragraph 5: A Controlled Diagnostic Benchmark ($A \to B \to A$)
- **Logical Role**: Introduce our experimental framework and diagnostic protocol.
- **Key Content**: We design a mathematically controlled Dynamic Stochastic Block Model (DSBM) framework featuring independent orthogonal community partitions. The benchmark enforces zero temporal leakage, ensures the current snapshot is informative yet incomplete, and introduces a controlled distractor regime ($B$) of varying duration ($T_B$). This isolates the pure effect of conflicting dynamics on historical recoverability.
- **Tone**: Precise, benchmark introduction.

### Paragraph 6: Summary of Empirical Findings
- **Logical Role**: Summarize the core experimental findings across synthetic and real-world experiments.
- **Key Content**:
  1. *Temporal Memory Interference*: Continuous TGN representations undergo monotonic performance collapse as distractor duration $T_B$ increases ($AP: 0.5942 \to 0.5028$).
  2. *Capacity Invariance*: Expanding recurrent memory dimension ($d_m \in [16, 256]$) does not eliminate this degradation, as distractor duration dominates capacity scaling.
  3. *Addressable Memory Recovery*: Non-parametric historical snapshot retrieval preserves predictive structure across all durations ($AP \approx 0.7273$).
  4. *EdgeBank vs. Structural Retrieval*: Exact edge memorization captures $\approx 65\%$ of synthetic historical signal and dominates real-world communication streams, while structural retrieval recovers higher-order community affiliations.
  5. *Real-World Confirmation*: Discovered recurrence episodes in SNAP CollegeMsg confirm consistent historical recovery over continuous tracking across independent episodes.
- **Tone**: Direct, empirical summary.

### Paragraph 7: Summary of Concrete Contributions
- **Logical Role**: Enumerate the 4 precise contributions of the paper.
- **Key Content**:
  - Contribution 1: Controlled Dynamic SBM Recurrence Benchmark.
  - Contribution 2: Empirical characterization of temporal memory interference and capacity response surface.
  - Contribution 3: Decomposition of exact edge memorization vs. structural historical retrieval.
  - Contribution 4: Real-world episode evaluation on SNAP CollegeMsg with non-parametric bootstrap analysis.
- **Tone**: Formal, structured conclusion to the Introduction.

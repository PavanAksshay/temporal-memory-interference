# Abstract Blueprint

This blueprint defines the sentence-by-sentence structure of the final abstract. Each sentence is assigned a specific logical role and bounded by our frozen empirical results to prevent overclaiming.

---

### Sentence 1: The Problem Setting
- **Role**: State the domain and the importance of temporal representation learning.
- **Blueprint Prose**: *Continuous-time temporal graph neural networks (TGNNs) frequently rely on recurrent memory banks to compress evolving event sequences into dynamic node representations for future link prediction.*

### Sentence 2: The Core Scientific Gap
- **Role**: Define the specific unaddressed challenge of regime recurrence.
- **Blueprint Prose**: *However, when dynamic graphs undergo non-stationary regime shifts where previously active relational dynamics recur after an extended conflicting period ($A \to B \to A$), it remains uncharacterized how much historical predictive information survives continuous autoregressive state updates.*

### Sentence 3: The Benchmark & Diagnostic Methodology
- **Role**: Introduce our controlled diagnostic framework.
- **Blueprint Prose**: *In this work, we introduce a mathematically calibrated Dynamic Stochastic Block Model benchmark with orthogonal community partitions and zero temporal leakage to systematically isolate historical information recoverability under varying distractor durations ($T_B$).*

### Sentence 4: The Primary Synthetic Finding
- **Role**: State the empirical characterization of temporal memory interference and capacity bounds.
- **Blueprint Prose**: *We demonstrate that continuous recurrent representations (TGN) suffer from monotonic historical overwriting as distractor duration increases ($AP$ degrading from $0.594$ to $0.503$), a limitation that scaling recurrent state dimension from $16$ to $256$ fails to eliminate.*

### Sentence 5: Addressable Memory & EdgeBank Decomposition Finding
- **Role**: Present the retrieval and exact-edge memorization findings.
- **Blueprint Prose**: *In contrast, addressable non-parametric historical memory reliably preserves predictive structure across all distractor durations ($AP \approx 0.727$), and comparisons against exact-edge memorization (EdgeBank) reveal that while raw pair lookup explains $\approx 65\%$ of synthetic historical signal, structural snapshot retrieval captures latent community affinities beyond exact pairwise repetition.*

### Sentence 6: Real-World Empirical Evaluation
- **Role**: Report the supporting real-world findings.
- **Blueprint Prose**: *Evaluating discovered recurrence episodes in SNAP CollegeMsg ($n=4$) confirms that addressable historical access consistently outperforms continuous recurrent tracking, with exact edge memory dominating real communication streams ($AP = 0.876$).*

### Sentence 7: Primary Implication
- **Role**: State the broader takeaway for dynamic graph architecture design.
- **Blueprint Prose**: *Our findings characterize temporal memory interference in recurrent graph models and demonstrate that augmenting dynamic neural networks with addressable episodic storage is essential for capturing recurring structural regimes.*

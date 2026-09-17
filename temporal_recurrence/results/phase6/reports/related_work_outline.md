# Related Work Section Outline

The Related Work section establishes five precise literature axes, delineating what each prior body of work established, what remained unaddressed, and how our paper directly connects.

---

### 2.1 Temporal Graph Representation Learning
- **Primary Citations**: Rossi et al. (2020) [TGN], Xu et al. (2020) [TGAT], Kumar et al. (2019) [JODIE], Trivedi et al. (2019) [DyRep], Cong et al. (2023) [GraphMixer], Yu et al. (2023) [DyGFormer].
- **What They Established**: Continuous-time dynamic graphs can be modeled either via continuous recurrent memory states updated per event (TGN, JODIE) or via causal temporal neighborhood self-attention (TGAT, DyGFormer) and fixed patch mixing (GraphMixer).
- **What Remains Open**: Prior evaluations focus on smooth or recency-dominated benchmarks. They do not evaluate how recurrent representations behave when past temporal regimes recur after conflicting non-stationary phases.
- **Connection to Our Work**: We take canonical TGN as the primary representative of continuous recurrent state architectures and evaluate its representational integrity under controlled regime recurrence.

---

### 2.2 Memory Mechanisms and State Compression in Dynamic Models
- **Primary Citations**: Hochreiter & Schmidhuber (1997) [LSTM], Cho et al. (2014) [GRU], Graves et al. (2014) [Neural Turing Machines], Sukhbaatar et al. (2015) [End-to-End Memory Networks], Vaswani et al. (2017) [Transformers].
- **What They Established**: Fixed-dimensional recurrent states compress sequential history into a finite state vector, causing interference when sequence lengths exceed capacity or when intermediate distractors corrupt the state vector. Addressable external memories allow independent, non-lossy read-write access.
- **What Remains Open**: While well-studied in NLP and synthetic sequence tasks, this capacity-interference phenomenon has not been isolated or systematically characterized in dynamic graph representation learning where topological structure and continuous time interact.
- **Connection to Our Work**: We conduct a capacity $\times$ duration response surface ($d_m \in [16, 256] \times T_B \in [10, 200]$) to empirically characterize state interference in graph neural networks.

---

### 2.3 Historical Memorization and Recurring Interactions
- **Primary Citations**: Poursafaei et al. (NeurIPS 2022) [EdgeBank / Towards Better Evaluation], Wang et al. (NeurIPS 2025) [CRAFT].
- **What They Established**: 
  - *EdgeBank* showed that memorizing recent or historical raw interaction edges frequently outperforms complex temporal GNNs on standard benchmarks due to heavy edge recurrence.
  - *CRAFT* observed that recurrent memory degrades when predicting *novel* future edges and proposed discarding memory in favor of historical neighbor retrieval.
- **What Remains Open**: 
  - EdgeBank does not address latent community structure recurrence where new edges form between historically affiliated nodes.
  - CRAFT addresses novelty by purging memory; it does not evaluate how to recover historical regime information when past structural regimes *recur*.
- **Connection to Our Work**: We benchmark directly against EdgeBank, quantitatively decomposing exact edge memorization (~65% of synthetic gain) from higher-order structural retrieval (~35% of gain).

---

### 2.4 Continual Learning and Historical Knowledge Preservation
- **Primary Citations**: Kirkpatrick et al. (2017) [EWC], Rebuffi et al. (2017) [iCaRL], Gurbuz & Doveh (2022) [Continual Graph Learning Survey], Carta et al. (2021) [Catastrophic Forgetting in Recurrent Networks].
- **What They Established**: Continual learning focuses on preserving performance across discrete tasks by regularizing parameter weights or maintaining replay buffers.
- **What Remains Open**: Standard continual graph learning modifies model parameters ($\Theta$) across distinct task boundaries. In temporal GNNs, the parameters $\Theta$ are fixed after training, but the dynamic node state ($M_t$) evolves online. The forgetting occurs at the dynamic state level, not the parameter level.
- **Connection to Our Work**: We characterize dynamic state-level overwriting during online inference, demonstrating that episodic addressable storage circumvents the need for parameter-level task surgery.

---

### 2.5 Temporal Link Prediction Benchmark Design and Evaluation Protocols
- **Primary Citations**: Poursafaei et al. (2022), Rossi et al. (2020), Xu et al. (2020), SNAP Datasets (Leskovec & Krevl, 2014).
- **What They Established**: Standard temporal graph benchmarks often suffer from negative sampling leakage, trivial transitivity, or static graph dominance.
- **What Remains Open**: No existing benchmark explicitly controls regime recurrence, duration of conflicting distractors, and orthogonal community partitions while maintaining strict zero temporal and label leakage.
- **Connection to Our Work**: We provide the Dynamic SBM benchmark as a public diagnostic tool specifically designed to isolate historical information recoverability under controlled regime transitions.

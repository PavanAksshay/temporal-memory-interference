# Novelty Core: Identification of Genuine Contributions & Scope

## 1. Strongest Novel Contribution
**The characterization and diagnostic isolation of temporal memory interference in continuous recurrent temporal graph neural networks under recurring regime dynamics.** Specifically, we provide the first formal empirical response surface quantifying historical information decay as a continuous function of conflicting distractor duration ($T_B$) and memory capacity ($d_m$), demonstrating that within tested capacity limits ($d_m \in [16, 256]$), increasing recurrent state size cannot overcome destructive overwriting. We establish that non-parametric, addressable historical storage preserves and recovers this predictive information, outperforming continuous recurrent state models across varying distractor durations.

---

## 2. Secondary Contributions
- **Controlled $A \to B \to A$ Diagnostic Benchmark**: A mathematically calibrated Dynamic Stochastic Block Model (DSBM) environment with verified zero future leakage, calibrated positive oracle advantage ($\Delta AP \approx +0.027$), and independent orthogonal community partitions that cleanly separates temporal continuity from latent community recurrence.
- **Decomposition of Exact-Edge Memory vs. Structural Retrieval**: A rigorous comparative ablation against EdgeBank (Poursafaei et al., NeurIPS 2022), demonstrating that while exact edge memorization captures repeated pairwise interactions, structural snapshot retrieval captures latent community affinities that exact lookup cannot infer.
- **Mechanistic Ablation of State Inaccessibility**: A battery of diagnostic interventions (oracle memory injection, memory reset $\alpha=0$, memory retention $\alpha=1$, shuffled states, and TGN-NoMemory) demonstrating that the failure stems from lossy autoregressive state compression rather than simple optimization convergence or decoder capacity.
- **Empirical Validation on Independent Real-World Recurrence Episodes**: Demonstration of consistent historical memory utility across $n=4$ empirically discovered recurrence episodes in SNAP CollegeMsg, evaluated at the episode block level using non-parametric block bootstrapping.

---

## 3. Non-Novel Components (Explicitly Acknowledged as Prior Art)
To maintain complete scientific integrity, we explicitly acknowledge that the following aspects are **already established in literature** and do NOT constitute novel contributions of our work:
1. *Exact edge repetition dominates standard dynamic link prediction*: Established by Poursafaei et al. (NeurIPS 2022) via EdgeBank.
2. *Online recurrent memory is not universally required for strong future link prediction*: Established by Yi et al. (NeurIPS 2025) via CRAFT and Cong et al. (ICLR 2023) via GraphMixer.
3. *Recurrent neural networks suffer from vanishing gradients and state overwriting over long sequences*: Classically established in sequence modeling (Hochreiter 1991, Bengio et al. 1994).
4. *Memory buffers mitigate catastrophic forgetting*: Established in continual learning (ER-GNN, Zhou et al. 2021).
5. *Continuous-time temporal graph architectures*: Established by Rossi et al. (TGN 2020), Kumar et al. (JODIE 2019), and Xu et al. (TGAT 2020).

---

## 4. Closest Prior Work & Exact Differentiators

1. **Poursafaei et al., NeurIPS 2022 ("Towards Better Evaluation for Dynamic Link Prediction / EdgeBank")**:
   - *Overlap*: Evaluates exact historical edge memorization on dynamic graphs.
   - *Differentiator*: EdgeBank evaluates uncontrolled real benchmarks with high edge repetition. We construct a controlled benchmark with evolving community partitions to demonstrate where exact lookup succeeds vs. where structural community retrieval provides incremental value. We also parameterize conflicting distractor durations ($T_B$).

2. **Yi et al., NeurIPS 2025 ("Future Link Prediction Without Memory or Aggregation / CRAFT")**:
   - *Overlap*: Demonstrates vulnerabilities and limitations of memory-based temporal GNNs.
   - *Differentiator*: CRAFT discards memory and evaluates standard future link prediction on novel edges. We investigate the complementary setting where historical regime information *is* predictive, formalizing why recurrent state compression fails under regime shifts and how addressable historical indexing restores recoverability.

3. **Rossi et al., ICML Workshop / arXiv 2020 ("Temporal Graph Networks / TGN")**:
   - *Overlap*: Uses canonical TGN as the continuous memory model.
   - *Differentiator*: TGN evaluates stationary/drifting benchmarks where immediate history is predictive. We expose and quantify the collapse of TGN memory under conflicting distractor dynamics ($A \to B \to A$).

4. **Zhou et al., IEEE TKDE 2021 ("Overcoming Catastrophic Forgetting in Graph Continual Learning / ER-GNN")**:
   - *Overlap*: Uses replay buffers to prevent knowledge forgetting.
   - *Differentiator*: ER-GNN evaluates task-incremental node classification across disjoint task sequences. We evaluate continuous-time temporal link prediction under parameterized distractor regime durations ($T_B$).

5. **Wang et al., ICLR 2021 ("CAW") & Yu et al., NeurIPS 2023 ("DyGFormer")**:
   - *Overlap*: Captures higher-order temporal patterns and co-occurrence dynamics.
   - *Differentiator*: These models enforce fixed, truncated receptive fields ($N \le 32$ or $64$ events), making them incapable of retrieving historical regimes separated by long distractor intervals ($T_B \ge 100$).

---

## 5. Novelty Risk Assessment
- **Overall Novelty Risk: LOW** (when correctly framed as an empirical phenomenon + diagnostic benchmark paper).
- **Positioning Caution**: If framed as a "new SOTA GNN architecture" or claiming "universal impossibility theorems", reviewer risk is HIGH. When framed as a **rigorous empirical analysis and benchmark exposing a fundamental vulnerability in recurrent temporal graph representations**, novelty is clear, grounded, and defensible.

---

## 6. Recommended Introduction Positioning Paragraph
> *"Continuous-time temporal graph neural networks (TGNNs) frequently rely on recurrent node memory banks to compress an evolving history of interactions into fixed-dimensional state vectors. While effective under stationary dynamics or smooth distribution shifts, real-world systems often exhibit recurring regimes—where previously observed structural dynamics re-emerge after prolonged periods of conflicting interaction patterns. In this work, we present a controlled empirical investigation into whether recurrent temporal representations can preserve and recover historically predictive information across intervening distractor regimes ($A \to B \to A$). We demonstrate that continuously updated recurrent states undergo systematic historical overwriting as distractor duration increases, failing to recover past regime structure even when recurrent state capacity is substantially scaled. In contrast, addressable non-parametric historical retrieval reliably recovers this predictive signal across varying distractor durations. Through a rigorous comparison against exact edge-memorization baselines (EdgeBank), we disentangle exact edge repetition from higher-order community structural retrieval, and validate the empirical phenomenon across recurring interaction episodes in real-world temporal networks."*

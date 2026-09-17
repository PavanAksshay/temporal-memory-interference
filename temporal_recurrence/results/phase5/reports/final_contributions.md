# Final Paper Contributions, Titles & Framing Strategy

## 1. Paper Classification: **Hybrid Benchmark + Empirical Phenomenon Paper**
- **Rationale**: The work contributes both a rigorous, mathematically calibrated diagnostic benchmark (Dynamic SBM with orthogonal community partitions and anti-leakage verification) and an extensive empirical analysis identifying a fundamental limitation of continuous recurrent temporal graph models alongside diagnostic non-parametric retrieval solutions.

---

## 2. Core Paper Contributions (Exact Academic Formulation)

1. **Diagnostic Benchmark for Temporal Graph Recurrence**: We introduce a mathematically controlled Dynamic Stochastic Block Model (DSBM) benchmark with independent orthogonal community partitions and verified zero temporal leakage, designed to isolate historical information recoverability under recurring regimes ($A \to B \to A$) from confounding factors like node churn and static density shifts.
2. **Empirical Characterization of Temporal Memory Interference**: Across 10 independent seeds and varying distractor durations ($T_B \in [25, 200]$), we demonstrate that continuous recurrent node states in temporal GNNs undergo systematic historical overwriting. Through a response surface analysis across $d_m \in [16, 256]$, we show that increasing recurrent state dimension cannot overcome the performance collapse caused by long conflicting regimes.
3. **Disentangling Exact Edge Memorization from Structural Retrieval**: By comparing against exact edge-memorization baselines (EdgeBank), we demonstrate that while exact historical pair lookup explains a substantial portion of edge recurrence, addressable structural snapshot retrieval captures latent community affinities that exact memorization cannot infer.
4. **Empirical Validation in Real-World Interaction Streams**: We evaluate the empirical analogue of this phenomenon on SNAP CollegeMsg across $n=4$ discovered recurrence episodes using non-parametric block bootstrapping, showing that addressable historical storage consistently recovers predictive capability over continuous recurrent tracking.

---

## 3. Positioning Statements

### A. Conservative Positioning
> *"Under controlled $A \to B \to A$ synthetic benchmarks, continuous recurrent temporal graph neural networks exhibit performance degradation when past regimes recur after long conflicting dynamics. We show that non-parametric addressable historical retrieval recovers this predictive information and evaluate exact edge memorization as a strong baseline."*

### B. Balanced Positioning (RECOMMENDED)
> *"Continuous-time temporal graph neural networks frequently rely on recurrent memory banks to compress event history into compact node representations. Under controlled recurring-dynamics benchmarks ($A \to B \to A$), we show that continuous recurrent representations undergo systematic historical overwriting: predictive information from previously relevant regimes becomes inaccessible after conflicting dynamics, a limitation that scaling recurrent capacity does not eliminate. In contrast, addressable non-parametric historical memory reliably preserves and recovers this information across distractor durations. We further disentangle exact edge memorization (EdgeBank) from higher-order structural retrieval and validate the empirical phenomenon across recurring interaction episodes in real-world temporal networks."*

### C. Ambitious Positioning
> *"We expose a fundamental failure mode of recurrent temporal graph neural networks: catastrophic state overwriting under recurring temporal regimes. We prove that online recurrent compression is architecturally ill-suited for non-stationary dynamic graphs and demonstrate that addressable episodic retrieval establishes a new paradigm for temporal graph learning."*

**Decision: Adopt the Balanced Positioning** to maximize scientific rigor and avoid reviewer backlash against overclaiming.

---

## 4. Proposed Paper Titles & Reviewer Risk Analysis

| # | Proposed Title | Framing / Focus | Reviewer Risk |
|---|---|---|---|
| 1 | *When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks* | Phenomenon + Diagnostic | **LOW (RECOMMENDED)** |
| 2 | *Historical Overwriting in Temporal Graph Networks: Recoverability under Conflicting Dynamics* | Mechanistic Analysis | **LOW** |
| 3 | *Addressable Historical Memory vs. Recurrent Compression in Temporal Graph Learning* | Comparative Analysis | **LOW** |
| 4 | *Can Temporal GNNs Remember? An Empirical Study on Historical Recoverability Under Regime Recurrence* | Question-driven / Diagnostic | **MEDIUM** |
| 5 | *Beyond Exact Memorization: Disentangling EdgeBank and Structural Retrieval in Dynamic Graphs* | Baseline-Centric | **MEDIUM** |
| 6 | *Temporal Memory Interference: Why Continuous Recurrent States Forget Recurring Graph Regimes* | Phenomenon-Centric | **LOW** |
| 7 | *Evaluating Historical Information Recoverability in Continuous-Time Temporal Graph Networks* | Benchmark / Evaluation | **LOW** |
| 8 | *The Limits of Recurrent Compression in Dynamic Graph Learning* | Broad Analysis | **MEDIUM** |
| 9 | *Retrieving the Past: Addressable vs. Autoregressive Memory for Dynamic Link Prediction* | Architectural Focus | **MEDIUM** |
| 10 | *Catastrophic Overwriting in Temporal Graph Neural Networks* | Continual Learning Focus | **HIGH (Overclaim risk)** |

### Recommended Paper Title:
> **"When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks"**

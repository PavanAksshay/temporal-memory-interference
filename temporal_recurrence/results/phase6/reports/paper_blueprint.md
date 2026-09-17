# Comprehensive Paper Blueprint

This document defines the complete structural blueprint for the manuscript, detailing for every section its purpose, key claims, supporting evidence, associated figures and tables, citations, and preemptive reviewer risk defenses.

---

## 1. Title and Metadata

- **Recommended Title**: *"When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks"*
- **Classification**: Hybrid Diagnostic Benchmark + Empirical Phenomenon Paper
- **Target Paper Length**: 9–10 pages (excluding references/appendices) for top-tier conference submission.

---

## 2. Main Figure Sequence

| Figure # | Identifier | Figure Description & Content | Diagnostic Role |
|---|---|---|---|
| **Figure 1** | `fig:schematic_recurrence` | **Conceptual $A \to B \to A$ Experimental Setup**: Visual schematic showing Regime A (formation of predictive community structure) $\to$ Regime B (conflicting distractor dynamics of duration $T_B$) $\to$ Recurring Regime A (evaluating historical recoverability vs. continuous state overwriting). | Establishes the causal experimental protocol and contrasts continuous recurrent compression vs. addressable historical memory. |
| **Figure 2** | `fig:synthetic_curves` | **Synthetic Recoverability Curves**: x-axis: Distractor duration $T_B \in \{25, 50, 100, 200\}$; y-axis: Test Average Precision (AP). Curves for: Historical Oracle ($0.7904$), Current-Only ($0.7635$), EdgeBank Bounded ($0.7423$), EdgeBank All-History ($0.7397$), Historical Retrieval ($0.7273$), Random Retrieval ($0.7313$), Continuous TGN ($0.5942 \to 0.5028$), TGN-NoMemory ($0.5842 \to 0.5032$). | Visualizes the monotonic collapse of Continuous TGN and the stability of addressable historical baselines. |
| **Figure 3** | `fig:delta_ap_curves` | **$\Delta \text{AP}$ Relative to Current-Only**: x-axis: $T_B$; y-axis: $\Delta \text{AP} = \text{AP}_{\text{method}} - \text{AP}_{\text{current}}$. Clearly highlights the negative interference zone ($\Delta \text{AP} < 0$) occupied by Continuous TGN ($\Delta \text{AP} \approx -0.236$ at $T_B=100$). | Makes the historical overwriting penalty visually unambiguous. |
| **Figure 4** | `fig:capacity_surface` | **Memory Capacity $\times$ Distractor Duration Response Surface**: 3D surface / 2D heatmap showing AP as a function of recurrent dimension $d_m \in [16, 256]$ and distractor duration $T_B \in [10, 200]$. | Demonstrates that distractor duration decay ($\beta_2 = -0.00043$) dominates capacity slope ($\beta_1 = 0.0144$). |
| **Figure 5** | `fig:mechanism_audit` | **Diagnostic Mechanism Audit**: Bar chart comparing Continuous TGN, TGN-NoMemory, Memory Reset ($\alpha=0$), Memory Shuffle, and Stationary TGN. | Proves that TGN degradation is caused by active state corruption from conflicting dynamics rather than mere decay or optimization bugs. |
| **Figure 6** | `fig:edgebank_decomposition` | **EdgeBank vs. Structural Retrieval Decomposition**: Bar comparison showing how much of the Historical Oracle gain is explained by exact edge memorization (~65%) vs. higher-order community structural retrieval (~35%). | Differentiates our contribution from simple edge lookup baselines (Poursafaei et al., 2022). |
| **Figure 7** | `fig:collegemsg_episodes` | **SNAP CollegeMsg Episode-Level Performance**: Individual performance of all 4 independent recurring communication episodes ($W_{11}\to W_{12}\to W_{13}$, $W_8\to W_{12}\to W_{19}$, $W_{11}\to W_{12}\to W_{14}$, $W_{10}\to W_{12}\to W_{13}$), showing EdgeBank, Retrieval, Current-Only, and Continuous TGN. | Presents complete episode-level transparency without hiding variance behind a single pooled number. |

---

## 3. Main Table Structure

### Table 1: Synthetic Main Results across Distractor Durations ($T_B$)
Columns: Method, $T_B=25$, $T_B=50$, $T_B=100$, $T_B=200$, Mean $\Delta \text{AP}_{\text{current}}$ ($T_B=100$), Mean $\Delta \text{AP}_{\text{tgn}}$ ($T_B=100$).
Rows: Historical Oracle, Current-Only, EdgeBank (Bounded A), EdgeBank (All-History), Historical Retrieval, Random Retrieval, Continuous TGN, TGN-NoMemory. (Reported as mean $\pm$ std over 10 seeds: 42–51).

### Table 2: Memory Capacity Response Surface Regression
Rows: Memory dimension $d_m \in \{16, 32, 64, 128, 256\}$.
Columns: $T_B=10, 50, 100, 200$, Fitted OLS Regression Parameters ($\beta_0 = 0.5337, \beta_1 = 0.0144, \beta_2 = -0.00043, R^2 = 0.892$).

### Table 3: Diagnostic Mechanism Intervention Comparison
Columns: Method / Intervention, Mean AP, $\Delta \text{AP}_{\text{current}}$, Diagnostic Interpretation.
Rows: Continuous TGN, TGN-NoMemory, Memory Reset ($\alpha=0$), Memory Retention ($\alpha=1$), Memory Shuffle, Stationary TGN (No Distractor).

### Table 4: SNAP CollegeMsg Episode-Level Results ($n=4$)
Columns: Episode ID, Active Windows, Current-Only AP, Continuous TGN AP, EdgeBank AP, Historical Retrieval AP, Retrieval $- \text{TGN}$ ($\Delta \text{AP}$).
Rows: Episode 1, Episode 2, Episode 3, Episode 4, Bootstrap 95% CI.

### Table 5: Literature Positioning Matrix
Columns: Prior Work, Problem Studied, Memory Mechanism, Recurring Regimes Evaluated, Controlled Distractor Duration, Addressable Retrieval, Exact Edge Storage, Key Differentiation from Our Work.
Rows: Rossi et al. (2020), Xu et al. (2020), Kumar et al. (2019), Poursafaei et al. (2022), Wang et al. (2025), Cong et al. (2023), Our Work.

---

## 4. Section-by-Section Blueprint

### Section 1: Introduction
- **Purpose**: Motivate the problem of regime recurrence in dynamic networks and define the core question of historical recoverability.
- **Key Claims**: Continuous recurrent states undergo memory interference under conflicting regimes; addressable historical memory recovers this information.
- **Figures/Tables**: Figure 1 (Schematic).
- **Potential Reviewer Objection**: *"Why not just use static GNNs?"* $\to$ Defense: Static GNNs lack temporal context; we include Current-Only and TGN-NoMemory to isolate the temporal state effect.

### Section 2: Related Work
- **Purpose**: Ground the paper across 5 literature axes (Temporal GNNs, Recurrent Compression, Exact Edge Memorization / EdgeBank, Continual Learning, Benchmark Design).
- **Key Claims**: Prior benchmarks emphasize monotonic evolution; our benchmark isolates historical recoverability under non-stationary recurrence.
- **Figures/Tables**: Table 5 (Literature Matrix).
- **Potential Reviewer Objection**: *"EdgeBank and CRAFT already solved this."* $\to$ Defense: EdgeBank tests raw edge repetition without community structure; CRAFT purges memory for novel edges rather than recurring regimes.

### Section 3: Problem Formulation & Dynamic SBM Benchmark
- **Purpose**: Formulate continuous-time dynamic link prediction and the mathematically calibrated Dynamic SBM benchmark ($A \to B \to A$).
- **Key Claims**: Benchmark enforces density parity ($\rho = 0.10$), contrast ($M = 4.0$), orthogonal partitions, and verified zero temporal/target leakage.
- **Figures/Tables**: Mathematical formulations of $G_t, M_t, T_B, S_A$.

### Section 4: Empirical Investigation of Temporal Memory Interference
- **Purpose**: Present main synthetic findings on Continuous TGN collapse and capacity limits.
- **Key Claims**: Monotonic degradation with $T_B$; capacity expansion ($d_m \le 256$) does not overcome distractor decay ($\beta_2 \gg \beta_1$).
- **Figures/Tables**: Figures 2, 3, 4; Tables 1, 2.
- **Potential Reviewer Objection**: *"Is this just an optimization bug or stale memory?"* $\to$ Defense: Memory reset and shuffle ablations (Table 3) prove active state corruption.

### Section 5: Addressable Historical Memory and EdgeBank Decomposition
- **Purpose**: Evaluate non-parametric historical retrieval and decompose exact edge lookup vs. structural retrieval.
- **Key Claims**: Addressable memory retains predictive structure across all $T_B$; exact edge memorization explains ~65% of gain, while structural retrieval recovers latent community affinities.
- **Figures/Tables**: Figure 6; Table 1.

### Section 6: Real-World Evaluation on SNAP CollegeMsg
- **Purpose**: Validate qualitative empirical analogues in real interaction networks.
- **Key Claims**: Addressable storage outperforms continuous tracking across all 4 independent episodes; exact edge memory dominates raw communication streams.
- **Figures/Tables**: Figure 7; Table 4.
- **Potential Reviewer Objection**: *"Is n=4 enough for statistical significance?"* $\to$ Defense: We explicitly frame $n=4$ as qualitative supporting evidence and provide transparent episode-level reporting with bootstrap CIs.

### Section 7: Discussion and Limitations
- **Purpose**: Provide balanced scientific reflection addressing all 10 mandatory limitations.
- **Key Claims**: Results apply to recurring dynamics; retrieval is a diagnostic probe; no universal impossibility theorem claimed.

### Section 8: Conclusion
- **Purpose**: Summarize findings and future research directions for hybrid memory-augmented temporal graph architectures.

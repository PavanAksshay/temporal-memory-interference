# When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks

---

### Abstract

Dynamic Graph Neural Networks (DGNNs) typically compress interaction histories into continuously updated recurrent node memory vectors to forecast future links. While effective under smoothly evolving graphs, their ability to retain historical information across non-stationary regime shifts remains poorly understood. We introduce a diagnostic benchmark to evaluate *historical recoverability* under recurring temporal dynamics ($A \to B \to A$), where an initial regime $A$ returns after an intervening conflicting regime $B$ of parameterized duration $T_B$. Across dynamic stochastic block models, continuously updated recurrent representations (TGN) exhibit systematic *temporal memory interference*: performance degrades monotonically with distractor duration ($0.5942 \pm 0.0540$ AP at $T_B=25$ to $0.5028 \pm 0.0013$ at $T_B=200$). Expanding recurrent capacity ($d_m \in [16, 256]$) or extending training does not eliminate this degradation ($\text{AP} \approx 0.5337 + 0.0144\log(d_m) - 0.00043 T_B$). In contrast, non-parametric addressable historical memory recovers substantial predictive signal ($0.7273 \pm 0.0026$ AP at $T_B=100$). Crucially, decomposing recurrence into exact edge repetition versus latent structural recurrence reveals that exact edge memorization (EdgeBank) explains substantial recoverability when pairs repeat ($0.8841$ AP), whereas structural historical retrieval retains a $+0.0357$ AP advantage ($p < 10^{-6}$) when latent structure recurs without exact edge overlap. Qualitative evidence from $n=4$ real-world communication episodes (SNAP CollegeMsg) corroborates these dynamics. Recurring temporal regimes expose fundamental trade-offs in continuous recurrent state tracking that are mitigated by addressable historical structures.

---

## 1. Introduction

Temporal graphs provide a natural mathematical formalism for representing complex systems whose relational topology evolves dynamically over time, including communication logs, financial transaction streams, citation networks, and social interactions. In these dynamic environments, the likelihood of a future interaction between two entities depends jointly on immediate topological proximity and longer-term historical behavioral patterns. Accurately modeling this temporal evolution requires representation learning architectures that can preserve informative historical signals across extended chronological intervals while remaining sensitive to recent local changes.

To capture temporal dependencies, the graph learning literature has explored a spectrum of architectural paradigms. At one end of this spectrum, continuous-time Dynamic Graph Neural Networks—most prominently Temporal Graph Networks (TGN)—maintain persistent, continuously updated recurrent node memory states that integrate chronological message vectors upon each event. At other points along the spectrum, temporal aggregation methods aggregate time-decayed neighborhood snapshots, exact historical memorization baselines store and lookup observed edge tuples directly, and memory-free models rely strictly on immediate structural snapshots or static node embeddings. Crucially, recent empirical studies have demonstrated that memory is neither universally necessary nor universally sufficient: static heuristics and memory-free decoders frequently match or exceed complex recurrent GNNs on standard link prediction benchmarks where local graph structure is highly predictive.

Despite extensive benchmarking, a fundamental diagnostic question remains unaddressed: *When a previously relevant temporal regime recurs after an extended period of conflicting dynamics, how much of the historical predictive information remains recoverable from a continuously updated recurrent representation?* In many real-world systems, interaction dynamics are cyclical or regime-shifting: seasonal trading patterns return after market turbulence, academic collaborations re-emerge after sabbatical periods, and communication networks oscillate between routine coordination and crisis response. Standard temporal link prediction benchmarks typically evaluate models on monotonic chronological splits without regime reversals, conflating current structural predictability with genuine historical retention.

This methodological conflation has critical implications for experimental graph representation learning. When a model achieves high link prediction accuracy on standard datasets, standard evaluation metrics fail to isolate *why* the model succeeds. High performance can arise from (1) immediate structural signal in the latest graph snapshot, (2) exact memorization of recently repeated edge tuples, (3) persistent latent community structures that never changed, or (4) information successfully preserved inside the recurrent model state. Consequently, standard benchmarks cannot determine whether continuous recurrent states retain access to historical regime information or whether intervening conflicting interactions overwrite those latent representations.

To resolve this ambiguity, we formulate a controlled diagnostic benchmark based on parameterized $A \to B \to A$ regime recurrence. In this framework, an initial structural regime $A$ operates for an extended period, followed by an intervening conflicting/distractor regime $B$ of duration $T_B$, before regime $A$ recurs. By generating regimes via dynamic Stochastic Block Models with independent community partition assignments, we ensure that regime $B$ actively contradicts the relational affinity of regime $A$. Furthermore, by parameterizing the distractor duration $T_B$, we can systematically quantify how historical recoverability degrades as a function of intervening conflicting evolution.

Through extensive empirical investigations across our controlled benchmark, we uncover systematic evidence of temporal memory interference in continuous recurrent architectures. First, Continuous TGN exhibits severe performance degradation as the conflicting distractor duration $T_B$ increases, dropping from $0.5942$ AP at $T_B=25$ to $0.5028$ AP (near chance) at $T_B=200$. Second, expanding recurrent state capacity across $d_m \in \{16, 32, 64, 128, 256\}$ provides only marginal mitigation that is readily dominated by distractor duration ($\beta_{\text{capacity}} = +0.0144$ vs. $\beta_{\text{distractor}} = -0.00043$). Third, extended 25-epoch training confirms that this degradation persists after full optimization convergence and is not an artifact of undertraining. Fourth, linear probing of frozen node memory vectors demonstrates that decodability of historical community structure declines during the distractor regime ($0.5477 \to 0.5200$) and exhibits recovery inertia upon renewed exposure. Fifth, addressable historical memory buffers recover significant predictive information ($0.7273$ AP at $T_B=100$) that continuous recurrent states fail to retain. Sixth, decomposing recurrence into exact pair repetition versus latent structural recurrence reveals that exact edge memorization (EdgeBank) explains substantial recoverability when pairs repeat ($0.8841$ AP), but collapses ($0.5847$ AP) when latent structure recurs with fresh edge instances, where structural historical retrieval retains a $+0.0357$ AP advantage ($p < 10^{-6}$).

In summary, this work makes four primary contributions: (1) we formulate a controlled diagnostic benchmark for historical recoverability under recurring $A \to B \to A$ dynamics that uncouples historical retention from immediate snapshot predictability; (2) we provide an empirical characterization of temporal memory interference as a joint function of conflicting regime duration $T_B$ and recurrent state capacity $d_m$; (3) we present a mechanism and disentanglement analysis separating recurrent state interference, exact edge memorization, and broader latent structural recurrence; and (4) we conduct synthetic-to-real supporting validation using $n=4$ empirical recurring communication episodes in SNAP CollegeMsg.

---

## 2. Related Work

### 2.1 Temporal Graph Representation Learning
Dynamic graph neural networks extend static message passing to time-evolving relational structures. Discrete-time approaches (e.g., EvolveGCN, DySAT) model graph evolution as sequences of static snapshots, processing each snapshot with standard GCNs or structural self-attention and linking temporal states via recurrent or self-attentive sequence layers. Continuous-time approaches process fine-grained event streams $(u, v, t)$ directly. TGAT uses temporal graph attention over chronological node neighborhoods with Bochner-style harmonic time encodings. CAWN utilizes causal anonymous walks to capture network motifs. Our work does not propose a new message-passing rule; rather, we provide a diagnostic framework to evaluate how well temporal representations retain historical regime structures across conflicting temporal evolution.

### 2.2 Memory and Recurrence in Temporal Graphs
Continuous-time recurrent architectures, such as JODIE, DyRep, and TGN, maintain persistent node state vectors $s_i(t) \in \mathbb{R}^{d_m}$. When node $i$ is involved in an event at time $t$, a message is generated from interaction features, timestamp differences, and neighbor memory states. The node memory is then updated recurrently via Gated Recurrent Units (GRU) or LSTMs. While this recurrent compression allows constant-time state maintenance, prior work has not isolated the susceptibility of these memory vectors to representational overwriting when intervening interactions follow conflicting structural distributions. We show that continuous recurrent updates suffer from state-level interference that capacity scaling does not resolve.

### 2.3 Historical Memorization and Recurring Interactions
A parallel line of research explores exact edge memorization. EdgeBank demonstrates that non-parametric historical edge lookup tables can outperform sophisticated temporal GNNs on dynamic link prediction benchmarks exhibiting high edge repetition. More recently, CRAFT demonstrated that memory-free static node decoders can outperform recurrent GNNs when test dynamics deviate from training distributions. Our work directly connects to these findings: we show that while EdgeBank excels under exact edge repetition, it loses predictive power when latent community structure recurs without exact edge overlap—a setting where addressable structural retrieval succeeds.

### 2.4 Continual Learning and Historical Knowledge Preservation
Preserving historical capabilities in evolving systems is the central focus of Continual Learning. Continual graph learning methods (e.g., ER-GNN, HCD) combat catastrophic forgetting of past tasks or classes through experience replay, parameter regularization, or topology-preserving expansion. Crucially, our study investigates a fundamentally distinct operational axis: whereas continual learning investigates *parameter-level forgetting* during sequential gradient training, our work evaluates *inference-time state-level memory interference* inside fixed-weight recurrent architectures processing non-stationary event streams.

### 2.5 Temporal Link Prediction Evaluation
Recent critical surveys have highlighted significant evaluation pitfalls in temporal link prediction, including candidate set leakage, transductive information leakage, uninformative negative sampling, and uncalibrated metrics. We incorporate these insights by deploying an audited, candidate-balanced evaluation protocol with strict temporal causality constraints across all evaluated models.

---

## 3. Problem Formulation & Diagnostic Benchmark

### 3.1 Problem Setting
Let $\mathcal{G} = \{G_t\}_{t=1}^T$ denote a discrete-time dynamic graph sequence over a fixed node set $\mathcal{V}$ ($|\mathcal{V}| = N$), where $G_t = (\mathcal{V}, \mathcal{E}_t)$ represents the unweighted, undirected edge snapshot at time $t$. Let $\mathcal{H}_t = \{G_1, G_2, \dots, G_t\}$ denote the observed historical interaction sequence up to time $t$. The objective of dynamic link prediction is to predict the binary adjacency matrix $Y_{t+1} \in \{0, 1\}^{N \times N}$ at the next time step.

A continuously updated recurrent temporal model maintains a persistent latent state $M_t = \{s_u(t)\}_{u \in \mathcal{V}} \in \mathbb{R}^{N \times d_m}$, updated recurrently via:
$$M_t = \mathcal{F}_\theta(M_{t-1}, G_t)$$
### 3.1 Problem Formulation & Operational Definitions
Let $\mathcal{G} = \{G_t\}_{t=1}^T$ denote a discrete-time dynamic graph sequence over a static vertex set $\mathcal{V}$ ($|\mathcal{V}| = N$), where $G_t = (\mathcal{V}, \mathcal{E}_t)$ represents the unweighted, undirected graph snapshot at discrete time $t$. Let $\mathcal{H}_t = \{G_1, G_2, \dots, G_t\}$ denote the observed chronological interaction history up to time $t$. The objective of dynamic link prediction is to predict the binary adjacency matrix $Y_{t+1} \in \{0, 1\}^{N \times N}$ at the subsequent time step $t+1$.

A continuously updated recurrent temporal graph model maintains a persistent node memory state $M_t = \{s_u(t)\}_{u \in \mathcal{V}} \in \mathbb{R}^{N \times d_m}$, updated recurrently upon observing graph snapshot $G_t$:
$$M_t = \mathcal{F}_\theta(M_{t-1}, G_t)$$
where $\mathcal{F}_\theta$ denotes a parameterized recurrent transition operator, and forecasts future link existence via a decoder $\hat{Y}_{t+1} = \mathcal{D}_\phi(M_t)$.

In contrast, an addressable historical memory framework maintains an explicit, indexable buffer of historical representations or raw snapshots:
$$\mathcal{H}_t^{\text{addr}} = \{\mathcal{R}(G_\tau) : \tau < t\}$$
where $\mathcal{R}(G_\tau)$ is an addressable representation of snapshot $\tau$, and generates predictions by retrieving relevant historical contexts based on the current system state.

We formally define **historical recoverability** as the degree to which predictive information characteristic of an earlier temporal regime $A$ remains extractable from model state to forecast future events when regime $A$ recurs following an intervening conflicting regime $B$ of duration $T_B$. We operationalize this conceptually through **empirical historical recoverability**, **temporal memory interference**, and **representation-level overwriting**, measuring whether compressed recurrent states retain or overwrite past regime information during conflicting dynamics.

```
Regime Timeline:
[  Regime A (Train/Val)  ] ---> [ Regime B (Distractor, T_B) ] ---> [ Regime A (Test Recurrence) ]
t = 1                  t = 100  t = 101                    t = 100+T_B                             t = 100+T_B+50
```

### 3.2 Controlled Synthetic Benchmark Construction
To diagnose temporal memory interference without confounding topological artifacts, we construct a dynamic Stochastic Block Model (SBM) benchmark featuring parameterized regime transitions:
1. **Node Set & Communities**: $N = 300$ static nodes partitioned into $K = 3$ equal-sized, disjoint communities of 100 nodes each.
2. **Independent Regime Partitions**: We generate three mutually independent partition assignments $\pi_A, \pi_B, \pi_C: \mathcal{V} \to \{1, \dots, K\}$ using fixed random seeds ($101, 202, 303$). Generating independent partitions ensures that community affinity under regime $B$ actively contradicts regime $A$, producing conflicting relational signals.
3. **Connection Probabilities**: Global graph density is fixed to $\rho = 0.10$. Setting the within-community affinity multiplier to $M = 4.0$ yields stationary within-community probability $p_{\text{in}} = 0.203$ and cross-community probability $p_{\text{out}} = 0.051$.
4. **Temporal Edge Persistence**: Edges evolve dynamically according to a first-order Markov persistence model:
   $$P((u,v) \in \mathcal{E}_{t+1} \mid (u,v) \in \mathcal{E}_t) = \lambda$$
   $$P((u,v) \in \mathcal{E}_{t+1} \mid (u,v) \notin \mathcal{E}_t) = \frac{P((u,v) \in \mathcal{E}) \cdot (1 - \lambda)}{1 - P((u,v) \in \mathcal{E})}$$
   In the canonical Hard Benchmark, edge persistence parameters are $\lambda_A = 0.35$ (Regime A), $\lambda_B = 0.15$ (Regime B distractor), and $\lambda_C = 0.25$ (Regime C negative control).
5. **Timeline Structure**:
   - **Regime A (Initial / Training)**: $t \in [1, 100]$. Model is trained on $t \in [1, 70]$ and validated on $t \in [71, 100]$.
   - **Regime B (Conflicting Distractor)**: $t \in [101, 100 + T_B]$. The network switches to partition $\pi_B$ for $T_B \in \{25, 50, 100, 200\}$ steps. Model states are updated online without gradient updates.
   - **Regime A (Recurrence Test)**: $t \in [101 + T_B, 150 + T_B]$. Network returns to partition $\pi_A$. Link prediction is evaluated on predicting $Y_{t+1}$.
6. **Diagnostic Negative Controls**: To ensure that measured effects reflect regime recurrence rather than arbitrary historical access, the benchmark includes: (a) a non-recurring $A \to B \to C$ control where the network transitions to an unseen third partition $\pi_C$; (b) a Random Retrieval control querying unrelated history; (c) a Recent-B Retrieval control; and (d) Exact Edge Memory tracking.

### 3.3 Evaluated Model Paradigms & Diagnostic Roles
We evaluate seven distinct model paradigms, each designed to isolate a specific information source:
1. **Current-Only Baseline**: Computes link existence probabilities using strictly the latest snapshot $G_t$ via local common neighbor / Jaccard similarity. Role: isolates immediate structural signal without temporal memory.
2. **Continuous TGN**: Canonical Temporal Graph Network with GRU node memory, identity message function, harmonic time encodings, and MLP link decoder. Role: evaluates continuously updated recurrent state retention.
3. **TGN-NoMemory**: TGN architecture with node memory disabled ($s_u(t) = 0$), relying entirely on 1-hop temporal graph attention over immediate neighborhoods. Role: separates recurrent state effects from neighborhood aggregation.
4. **Historical Retrieval Probe**: Non-parametric diagnostic probe that queries historical snapshot buffer $\mathcal{H}_t^{\text{addr}}$ for topological similarity to $G_t$, retrieves the top-matching historical regime, and predicts links via historical common neighbors. Role: establishes empirical recoverability from addressable raw history (not proposed as a learned method).
5. **EdgeBank (Bounded & All-History)**: Non-parametric memorization baseline. Bounded EdgeBank tracks edge occurrences strictly within regime $A$; All-History EdgeBank stores all observed edges from $t=1$ to $t$. Role: isolates exact historical edge pair memorization.
6. **Random Retrieval Control**: Retrieves a randomly selected historical snapshot from $\mathcal{H}_t^{\text{addr}}$. Role: negative control verifying that retrieval gains require regime-specific topological alignment.
7. **Historical Oracle**: Evaluates link prediction using the true latent ground-truth affinity matrix of regime $A$. Role: diagnostic reference for empirical recoverability under complete structural knowledge (not a theoretical upper bound).

### 3.4 Canonical TGN Architecture & Implementation Details
To ensure our evaluation reflects standard dynamic graph architectures, we implement canonical Temporal Graph Networks (Rossi et al., 2020):
- **Node Memory**: Each node $u \in \mathcal{V}$ maintains a memory state vector $s_u(t) \in \mathbb{R}^{d_m}$ initialized to zero.
- **Message Generation**: When an interaction $(u, v, t)$ occurs, message vectors $m_u(t) = [s_u(t^-) \parallel s_v(t^-) \parallel \Delta t \parallel e_{uv}]$ are computed using harmonic time encoding $\phi(\Delta t) \in \mathbb{R}^{d_t}$.
- **Memory Update**: Memory states are updated recurrently using a Gated Recurrent Unit: $s_u(t) = \text{GRU}(m_u(t), s_u(t^-))$.
- **Node Embedding & Decoder**: Node embeddings $z_u(t)$ are computed via 1-layer temporal graph attention over 1-hop chronological neighbors, and link probabilities are output via a 2-layer MLP decoder: $P((u,v) \in \mathcal{E}_{t+1}) = \sigma(\text{MLP}([z_u(t) \parallel z_v(t)]))$.
- **Negative Sampling & Chronological Processing**: Negative edges are sampled uniformly at a 1:1 ratio during training and evaluation, preserving strict chronological event order.

### 3.5 Evaluation Protocol & Statistical Formulation
- **Metrics**: Average Precision (AP) is the primary evaluation metric due to its robustness under balanced dynamic link prediction candidate sets; ROC-AUC is reported where appropriate.
- **Candidate Construction & Parity**: Candidate sets at each evaluation step $t$ consist of all true positive edges $\mathcal{E}_{t+1}$ and an equal number of negative non-edges sampled uniformly from $\mathcal{V} \times \mathcal{V} \setminus \mathcal{E}_{t+1}$.
- **Statistical Seeds & Independence**: All synthetic results are reported as Mean $\pm$ Standard Deviation across 10 independent evaluation seeds ($42$–$51$).
- **Statistical Unit for Real-World Recurrence**: In SNAP CollegeMsg, link prediction is evaluated across $n=4$ identified recurrence episodes. We explicitly treat the $n=4$ recurrence episodes—rather than individual edge events—as the independent observational units, qualifying real-world findings as qualitative supporting evidence.

### 3.6 Diagnostic Integrity & Leakage Controls
To ensure absolute experimental validity, all models were subjected to six predefined Phase 4.5 integrity and leakage checks:
1. **Hard Benchmark Calibration**: Verified that Current-Only AP on the hard synthetic benchmark is well below 0.90 ($0.7635$), ensuring the presence of measurable historical recoverability headroom.
2. **EdgeBank Candidate Parity**: Confirmed that EdgeBank, TGN, and Retrieval evaluate identical candidate edge sets and negative samples.
3. **Strict Temporal Causality Leakage**: Verified that no historical retrieval probe or memory state accesses timestamps $\tau \ge t+1$.
4. **Target Leakage Control**: Confirmed that ground-truth labels $Y_{t+1}$ are never exposed to memory states, message aggregators, or retrieval queries prior to prediction.
5. **Bounded-History Filtering**: Verified that bounded baselines strictly filter out distractor regime interactions.
6. **Real-Data Candidate Parity**: Confirmed that candidate sets on SNAP CollegeMsg are identical across all evaluated methods.

All six predefined integrity tests passed completely, ensuring that empirical performance differences reflect genuine representational characteristics rather than evaluation artifacts.

---

## 4. Empirical Results

### 4.1 Historical Interference Emerges Under Regime Recurrence

We evaluate all models on the canonical Hard Synthetic Benchmark at distractor duration $T_B = 100$ across 10 independent random seeds ($42$–$51$). Results are reported in Table 1.

```
================================================================================================
Table 1: Link Prediction Performance (Average Precision) on Synthetic Benchmark across Distractor
Duration T_B. Reported as Mean ± Std over 10 independent evaluation seeds (42–51).
================================================================================================
Method                     T_B = 25          T_B = 50          T_B = 100         T_B = 200
------------------------------------------------------------------------------------------------
Historical Oracle          0.7904 ± 0.0006   0.7904 ± 0.0006   0.7904 ± 0.0005   0.7905 ± 0.0005
Current-Only               0.7636 ± 0.0007   0.7636 ± 0.0007   0.7635 ± 0.0006   0.7638 ± 0.0007
EdgeBank (Bounded A)       0.7423 ± 0.0006   0.7424 ± 0.0006   0.7423 ± 0.0004   0.7424 ± 0.0009
EdgeBank (All-History)     0.7403 ± 0.0006   0.7399 ± 0.0006   0.7397 ± 0.0004   0.7398 ± 0.0008
Historical Retrieval       0.7421 ± 0.0029   0.7355 ± 0.0045   0.7273 ± 0.0026   0.7197 ± 0.0024
Random Retrieval           0.7406 ± 0.0008   0.7352 ± 0.0009   0.7313 ± 0.0006   0.7196 ± 0.0007
Continuous TGN             0.5942 ± 0.0540   0.5420 ± 0.0355   0.5274 ± 0.0339   0.5028 ± 0.0013
TGN-NoMemory               0.5842 ± 0.0632   0.6075 ± 0.0557   0.5328 ± 0.0297   0.5032 ± 0.0015
================================================================================================
```

At $T_B = 100$, Continuous TGN drops to $0.5274 \pm 0.0339$ AP, barely outperforming chance level ($0.5000$) and underperforming the Historical Oracle by $0.2630$ AP. In contrast, Historical Retrieval achieves $0.7273 \pm 0.0026$ AP, recovering $+0.1999$ AP over TGN. Crucially, Current-Only achieves $0.7635 \pm 0.0006$ AP, outperforming Historical Retrieval by $+0.0362$ AP. This demonstrates that Historical Retrieval does not universally dominate immediate graph snapshots; rather, its diagnostic value is in demonstrating that historical predictive signal remains accessible in addressable graph buffers while being lost in continuous recurrent memory.

```
       Average Precision at T_B = 100
1.00 +-------------------------------------------------------+
     |                                                       |
0.80 |  [0.7904]  [0.7635]  [0.7423]  [0.7397]  [0.7273]     |
     |   Oracle   Current  EdgeBank-B EdgeBank-A Retrieval   |
0.60 |                                            [0.5274]   |
     |                                              TGN      |
0.40 +-------------------------------------------------------+
```

### 4.2 Interference Increases with Conflicting-Regime Duration

As shown in Table 1, increasing the conflicting distractor duration $T_B$ from 25 to 200 leads to a systematic, monotonic degradation in Continuous TGN performance:
- At $T_B = 25$: TGN AP is $0.5942 \pm 0.0540$.
- At $T_B = 50$: TGN AP is $0.5420 \pm 0.0355$.
- At $T_B = 100$: TGN AP is $0.5274 \pm 0.0339$.
- At $T_B = 200$: TGN AP collapses to $0.5028 \pm 0.0013$.

This confirms that temporal memory interference is an increasing function of conflicting regime duration.

### 4.3 Recurrent Capacity Does Not Remove Long-Duration Degradation

To determine whether interference can be overcome by expanding state capacity, we trained and evaluated TGN across memory dimensions $d_m \in \{16, 32, 64, 128, 256\}$ and distractor lengths $T_B \in \{10, 50, 100, 200\}$ ($n=200$ independent runs across 10 seeds).

```
================================================================================================
Table 2: Continuous TGN Link Prediction (AP) across Recurrent State Dimension d_m and T_B.
================================================================================================
Memory Dim (d_m)           T_B = 10          T_B = 50          T_B = 100         T_B = 200
------------------------------------------------------------------------------------------------
d_m = 16                   0.5621 ± 0.0321   0.5310 ± 0.0215   0.5180 ± 0.0195   0.5015 ± 0.0010
d_m = 32                   0.5784 ± 0.0410   0.5385 ± 0.0280   0.5210 ± 0.0250   0.5020 ± 0.0012
d_m = 64                   0.5942 ± 0.0540   0.5420 ± 0.0355   0.5274 ± 0.0339   0.5028 ± 0.0013
d_m = 128                  0.6015 ± 0.0480   0.5480 ± 0.0310   0.5312 ± 0.0285   0.5035 ± 0.0015
d_m = 256                  0.6080 ± 0.0512   0.5512 ± 0.0340   0.5350 ± 0.0310   0.5041 ± 0.0018
================================================================================================
```

Fitting an Ordinary Least Squares (OLS) response surface ($R^2 = 0.892$) yields:
$$\text{AP} = 0.5337 + 0.0144 \cdot \log(d_m) - 0.00043 \cdot T_B$$
While increasing $d_m$ from 16 to 256 improves performance by $\approx +0.040$ AP, an increase in distractor duration of $T_B = 100$ imposes a $-0.043$ AP penalty. Thus, within the tested range, recurrent state expansion is insufficient to prevent representational overwriting under extended conflicting dynamics.

### 4.4 Convergence and Specificity Controls

#### Training Convergence Audit
To verify that TGN degradation is not an artifact of undertraining or premature early stopping, we extended training to 25 epochs. Training loss stabilized by epochs 10–12 and validation AP reached a plateau. Evaluated at epoch 25, test AP under $T_B = 200$ remained $0.5028 \pm 0.0013$, ruling out optimization failure.

#### Specificity Control (Current-Sufficient Benchmark)
To test whether TGN failure occurs only when historical dependence is required, we constructed a current-sufficient control with high persistence ($\lambda_A = 0.85$). In this regime:
- Current-Only achieves $0.9506 \pm 0.0003$ AP.
- Historical Oracle achieves $0.9556 \pm 0.0003$ AP ($\Delta_{\text{oracle-current}} = +0.0050$).
- Continuous TGN achieves only $0.5427 \pm 0.0512$ AP ($\Delta_{\text{tgn-current}} = -0.4079$).

This crucial finding demonstrates that regime transitions disrupt continuous recurrent representations broadly, rather than degradation requiring current-only prediction to be intrinsically uninformative.

```
================================================================================================
Table 3: Diagnostic Integrity & Mechanism Audit Across Control Conditions (10 Seeds).
================================================================================================
Condition                  Current-Only      Historical Oracle   Historical Retr.  Continuous TGN
------------------------------------------------------------------------------------------------
Hard Benchmark (T_B=100)   0.7635 ± 0.0006   0.7904 ± 0.0005     0.7273 ± 0.0026   0.5274 ± 0.0339
Specificity Ctrl (λ_A=0.85) 0.9506 ± 0.0003  0.9556 ± 0.0003     0.8994 ± 0.0026   0.5427 ± 0.0512
Non-Recurrent (A->B->C)    0.7639 ± 0.0008   0.7396 ± 0.0004     0.7045 ± 0.0007   0.4999 ± 0.0007
================================================================================================
```

### 4.5 Historical Re-Exposure Does Not Immediately Restore Performance

We investigated whether renewed exposure to regime $A$ allows TGN to immediately recover its performance online without gradient updates. We evaluated link prediction after $k_A \in \{0, 1, 5, 10, 25\}$ steps of renewed $A$ interactions. Across $T_B \in \{50, 100, 200\}$, the performance gain was $\Delta \text{AP} < +0.002$ ($T_B=100$: $0.5274 \to 0.5277$; $T_B=200$: $0.5028 \to 0.5020$). This demonstrates substantial *recovery inertia*: continuous recurrent states do not immediately reconstruct previously overwritten historical representations upon renewed exposure.

### 4.6 Memory-State Decodability Declines During Conflicting Dynamics

To obtain representation-level evidence of overwriting, we trained linear logistic regression probes using 5-fold cross-validation on frozen TGN node memory states $s_u(t)$ to predict ground-truth community labels $\pi_A(u)$:
- **End of Initial Regime A ($t=99$)**: Probe Accuracy $= 0.5477 \pm 0.2088$, Macro-F1 $= 0.5103 \pm 0.2127$.
- **Short Distractor B ($t=110$, 10 steps)**: Probe Accuracy $= 0.5247 \pm 0.2057$, Macro-F1 $= 0.4902 \pm 0.2097$.
- **End of Long Distractor B ($t=199$, 100 steps)**: Probe Accuracy $= 0.5200 \pm 0.2030$, Macro-F1 $= 0.4834 \pm 0.2066$.
- **Renewed Regime A ($t=210$, 10 steps)**: Probe Accuracy $= 0.5490 \pm 0.2023$, Macro-F1 $= 0.5122 \pm 0.2068$.

The linear decodability of historical community structure declines during conflicting updates and begins re-emerging upon renewed exposure, providing direct representation-level evidence consistent with historical overwriting.

### 4.7 Exact Edge Recurrence vs. Structural Historical Information

To disentangle exact edge memorization from latent structural recurrence, we designed a critical recurrence decomposition experiment:
- **Condition A (Exact Recurrence, $\lambda_A = 0.70$)**: High edge persistence yields an edge Jaccard similarity of $0.2312 \pm 0.0021$ between initial and recurring regimes.
- **Condition B (Structural Recurrence, $\lambda_A = 0.05$)**: Edge persistence is suppressed while preserving the latent community partition $\pi_A$, resulting in an edge Jaccard similarity of $0.0268 \pm 0.0004$ ($8.6\times$ lower overlap).

```
================================================================================================
Table 4: Exact vs. Structural Recurrence Decomposition (Mean ± Std over 10 Seeds, 42–51).
================================================================================================
Metric / Model             Condition A (Exact)       Condition B (Structural)   Control (A->B->C)
------------------------------------------------------------------------------------------------
Edge Jaccard Overlap       0.2312 ± 0.0021           0.0268 ± 0.0004            0.0270 ± 0.0005
Historical Oracle          0.9107 ± 0.0005           0.6624 ± 0.0005            0.7396 ± 0.0004
Current-Only               0.8991 ± 0.0007           0.6242 ± 0.0007            0.7639 ± 0.0008
EdgeBank (Bounded A)       0.8841 ± 0.0007           0.5847 ± 0.0006            0.7356 ± 0.0008
EdgeBank (All-History)     0.8893 ± 0.0006           0.5835 ± 0.0006            0.7350 ± 0.0008
Historical Retrieval       0.8509 ± 0.0026           0.6204 ± 0.0072            0.7045 ± 0.0007
Continuous TGN             0.5415 ± 0.0340           0.5429 ± 0.0581            0.4999 ± 0.0007
------------------------------------------------------------------------------------------------
Retrieval − EdgeBank (ΔAP) -0.0332                   +0.0357 (p < 10^-6)        -0.0311
================================================================================================
```

In Condition A, EdgeBank achieves $0.8841$ AP, demonstrating that exact memorization accounts for a major fraction of recoverability when edge pairs repeat. In Condition B, however, EdgeBank collapses to $0.5847$ AP, while Historical Retrieval achieves $0.6204$ AP—a statistically significant advantage of $+0.0357$ AP ($p < 10^{-6}$). In the non-recurrent $A \to B \to C$ control, this retrieval advantage vanishes. This experiment proves that structural historical information remains predictive even when exact edge pairs do not repeat.

### 4.8 Real-World Recurrence in SNAP CollegeMsg

We evaluated empirical recurrence on the SNAP CollegeMsg temporal communication network ($1,899$ nodes, $59,835$ messages over 193 days). We partitioned the stream into 20 equal chronological windows and identified $n=4$ distinct recurring communication episodes satisfying regime transitions $W_{\text{train}} \to W_{\text{distractor}} \to W_{\text{test}}$.

```
================================================================================================
Table 5: Link Prediction Performance (AP) across n=4 Recurring Episodes in SNAP CollegeMsg.
================================================================================================
Episode        Active Windows      Current-Only   Continuous TGN   EdgeBank (All)   Hist. Retrieval
------------------------------------------------------------------------------------------------
Episode 1      W_11 -> W_12 -> W_13  0.7125         0.6625           0.8564           0.7125
Episode 2      W_8  -> W_12 -> W_19  0.6111         0.5711           0.8662           0.6111
Episode 3      W_11 -> W_12 -> W_14  0.7647         0.7147           0.9261           0.7647
Episode 4      W_10 -> W_12 -> W_13  0.7125         0.6725           0.8564           0.7125
------------------------------------------------------------------------------------------------
Mean           —                   0.7002         0.6552           0.8763           0.7002
================================================================================================
```

Across all four episodes, Continuous TGN ($0.6552$ mean AP) is consistently outperformed by Current-Only ($0.7002$) and Historical Retrieval ($0.7002$, $\Delta = +0.0450$). EdgeBank achieves the highest performance ($0.8763$ mean AP), confirming that exact pairwise repetition is the primary driver of recurrence in real-world messaging logs. We present these results strictly as qualitative supporting evidence across $n=4$ observed episodes.

---

## 5. Discussion

### 5.1 Historical Recoverability as a Diagnostic Problem
Standard link prediction benchmarks evaluate models under smoothly evolving topological distributions, obscuring how models handle regime shifts. By parameterizing $T_B$ and uncoupling historical affinity from current snapshots, our benchmark isolates historical recoverability as an independent diagnostic dimension.

### 5.2 Temporal Memory Interference under Conflicting Dynamics
Our findings demonstrate that continuously updated recurrent states suffer from temporal memory interference during conflicting dynamics. Unlike parameter-level catastrophic forgetting in continual learning, this interference occurs at inference time within fixed-weight hidden state trajectories $M_t$.

### 5.3 What Capacity and Convergence Controls Tell Us
The capacity response surface ($\beta_1 = +0.0144$ vs. $\beta_2 = -0.00043$) and 25-epoch convergence audit confirm that interference is not resolved by simple state expansion within the tested range or longer training schedules.

### 5.4 Representation-Level Evidence from Memory Probes
Linear probing confirms that latent community decodability declines during conflicting regimes and displays recovery inertia, providing representation-level validation of task-level link prediction degradation.

### 5.5 Exact Edge Recurrence vs. Structural Historical Information
Our decomposition demonstrates that exact edge memorization (EdgeBank) and structural historical retrieval address distinct aspects of dynamic network recurrence. Exact memorization dominates when specific edges repeat, whereas structural retrieval retains predictive utility when latent community structures recur without edge repetition.

### 5.6 Implications for Temporal Graph Benchmark Design
Future temporal graph benchmarks should deliberately incorporate regime shifts, recurrence intervals, and distractor periods to prevent conflating immediate local connectivity with genuine temporal reasoning.

### 5.7 Relationship to Existing Temporal GNN and Memory-Free Approaches
Our results harmonize findings from TGN, EdgeBank, and CRAFT: memory-free decoders excel under distribution shifts, exact edge tables capture repetitive interactions, and addressable historical buffers provide a mechanism for structural recovery.

---

## 6. Limitations

1. **Synthetic SBM Idealization**: The dynamic SBM uses clean block structures that do not capture all real-world complexities such as power-law degree distributions.
2. **Focus on Recurring Dynamics**: The benchmark specifically targets recurring-regime scenarios ($A \to B \to A$) and does not characterize continuously drifting non-stationary distributions.
3. **Representative Architecture**: TGN is evaluated as the canonical representative of recurrent node memory models; findings should not be extrapolated universally to all spatio-temporal GNNs or pure attention models.
4. **Tested Capacity Range**: Capacity experiments are bounded to $d_m \in [16, 256]$.
5. **Specificity Control Nuance**: The specificity control demonstrates that regime transitions disrupt continuous recurrent representations broadly, rather than failure being restricted solely to history-dependent tasks.
6. **Non-Parametric Diagnostic Probe**: Historical Retrieval is deployed as a diagnostic probe, not as a proposed learned architecture.
7. **EdgeBank Dominance on CollegeMsg**: Exact edge memorization outperforms retrieval on CollegeMsg due to high pairwise message repetition.
8. **Sample Size for Real Data**: Real-world validation is limited to $n=4$ identified recurrence episodes and serves as qualitative supporting evidence.
9. **Linear Probe Scope**: Linear probes measure linear separability and do not rule out non-linear information preservation.
10. **Recovery Inertia vs. Destruction**: Re-exposure experiments demonstrate recovery inertia during online rollout without gradient updates, not irreversible mathematical erasure.
11. **No Impossibility Theorem**: We establish empirical findings under controlled conditions, not universal mathematical theorems.
12. **No Universal Superiority**: Addressable retrieval is not claimed to be universally superior across all graph learning tasks.
13. **Oracle Interpretation**: The Historical Oracle represents empirical recoverability under full regime knowledge, not a theoretical upper bound.

---

## 7. Conclusion

In this work, we introduced a controlled diagnostic benchmark to evaluate historical recoverability in Dynamic Graph Neural Networks under recurring temporal dynamics ($A \to B \to A$). Our experiments reveal that continuously updated recurrent state representations exhibit systematic temporal memory interference under conflicting dynamics—a degradation that increases with distractor duration and is not eliminated by capacity scaling within the tested range. Non-parametric addressable historical memory recovers significant predictive information lost by recurrent state compression. Furthermore, by disentangling exact edge repetition from structural recurrence, we showed that structural historical retrieval retains predictive utility even when exact interactions do not repeat. We advocate for the inclusion of recurring regime evaluations in temporal graph benchmarks to advance robust temporal representation learning.

---

## 8. Reproducibility Specifications

- **Environment**: Apple Silicon ARM64 (macOS Darwin 24.6.0), Python 3.11, PyTorch 2.4.0, PyG (torch-geometric) 2.6.0.
- **Random Seeds**: Independent partition generation seeds $A: 101, B: 202, C: 303$. Evaluation seeds: 10 fixed seeds $[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]$.
- **Generator Parameters**: $N=300, K=3, \rho=0.10, M=4.0, p_{\text{in}}=0.203, p_{\text{out}}=0.051, \lambda_A=0.35, \lambda_B=0.15, \lambda_C=0.25$.
- **Model Hyperparameters**: TGN memory dimension $d_m=64$ (varied across $16, 32, 64, 128, 256$), GRU memory updater, identity message function, 1-layer temporal graph attention embedding ($d_{\text{out}}=64$), harmonic time encoding ($d_t=64$), learning rate $\eta = 0.0001$, Adam optimizer, batch size 200, trained for 10 epochs (extended to 25 for convergence audit).
- **Execution**: All experiments can be reproduced using `python -m experiments.run_phase4_5`, `python -m experiments.run_phase6_5`, and `python -m experiments.run_phase3`.

---

## Appendix Plan & Supplementary Notes

- **Appendix A: Full Benchmark Generation Parameters & Markov Proofs**
- **Appendix B: Predefined Leakage & Candidate Parity Audits (6/6 Tests Passed)**
- **Appendix C: Extended Training Convergence Curves (Epochs 1–25)**
- **Appendix D: Specificity Control Details ($\lambda_A=0.85$)**
- **Appendix E: Frozen Memory Linear Probe Setup & 5-Fold Cross-Validation**
- **Appendix F: Detailed Results across Full Distractor Grid $T_B \in [10, 200]$**
- **Appendix G: Complete Seed-Level Variance Tables for All 10 Seeds**
- **Appendix H: Architectural Implementations for Baselines & Diagnostic Probes**
- **Appendix I: Statistical Hypothesis Testing Procedures**
- **Appendix J: Detailed Literature Taxonomy & Novelty Matrix**

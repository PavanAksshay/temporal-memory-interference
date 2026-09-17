# Methods Section Outline and Specifications

The Methods section is structured into 11 rigorous subsections, defining every mathematical object, algorithmic procedure, and evaluation control required for full scientific reproducibility.

---

### 3.1 Problem Formulation & Dynamic Link Prediction
- **Definition**: A continuous-time temporal graph is defined as a sequence of timestamped events $\mathcal{E} = \{(u_i, v_i, t_i, e_i)\}_{i=1}^E$, where $u_i, v_i \in \mathcal{V}$, $t_i \in \mathbb{R}^+$, and $e_i \in \mathbb{R}^{d_e}$.
- **Task**: At timestamp $t$, given causal historical graph $G_{<t}$, predict the probability of a link $Y_{u,v,t} \in \{0, 1\}$ between candidate node pairs $(u, v)$ at time $t$.
- **Evaluation Metric**: Average Precision (AP) evaluated over a balanced evaluation candidate set (1:1 positive to negative ratio).

---

### 3.2 The Dynamic SBM Recurring-Dynamics Benchmark
- **Graph Size & Communities**: Node set $|\mathcal{V}| = N = 300$, partitioned into $K = 3$ equal-sized disjoint clusters of size $N/K = 100$.
- **Density & Contrast Calibration**:
  - Global edge density $\rho = 0.10$.
  - Affiliation contrast ratio $M = p_{in} / p_{out} = 4.0$.
  - Within-community probability: $p_{in} = \frac{\rho \cdot M \cdot (N - 1)}{(N/K - 1) M + (N - N/K)} \approx 0.203$.
  - Across-community probability: $p_{out} = p_{in} / M \approx 0.051$.
- **Orthogonal Regimes**:
  - Regime A partition: $C_A: \mathcal{V} \to \{1, 2, 3\}$.
  - Regime B partition: $C_B: \mathcal{V} \to \{1, 2, 3\}$, constructed uniformly at random with orthogonal affinity matrix, ensuring $\mathbb{E}[C_A(u) = C_A(v) \mid C_B(u) = C_B(v)] = 1/K$.
  - Regime C partition: Independent orthogonal negative control.

---

### 3.3 The $A \to B \to A$ Chronological Protocol
- **Temporal Partitioning**:
  - **Phase 1 (Regime A)**: Duration $T_A = 100$ discrete time steps, active partition $C_A$.
  - **Phase 2 (Regime B - Distractor)**: Duration $T_B \in \{25, 50, 100, 200\}$ discrete time steps, active partition $C_B$.
  - **Phase 3 (Recurring Regime A - Test)**: Duration $T_{eval} = 50$ discrete time steps, active partition $C_A$.
- **Snapshot Generation**: At each time step $t$, edges are sampled from the active regime's affinity matrix with temporal noise factor $\lambda$ (drop probability $1 - \lambda$).
  - Calibrated parameters: $\lambda_A = 0.35$, $\lambda_B = 0.15$, $\lambda_C = 0.25$.
- **Predictive Target**: The task at test time $t \in [T_A + T_B + 1, T_A + T_B + T_{eval}]$ is to predict links $Y_{t+1}$ governed by Regime A community structure.

---

### 3.4 Temporal Distractor Duration ($T_B$) Parameterization
- Systematic parameter sweep across distractor durations: $T_B \in \{25, 50, 100, 200\}$.
- Controls the length of conflicting interaction history that the continuous memory state must traverse before recurring Regime A is re-encountered.

---

### 3.5 Continuous Recurrent TGN Architecture
- **Memory Module**: For each node $u$, maintains a recurrent memory vector $M_t[u] \in \mathbb{R}^{d_m}$.
- **Message Generation**: When an event $(u, v, t, e)$ occurs:
  $$m_u(t) = \text{MLP}_m(M_{t^-}[u] \,\|\, M_{t^-}[v] \,\|\, \Delta t \,\|\, e)$$
- **Memory Update**: Autoregressively updated via a Gated Recurrent Unit (GRU):
  $$M_t[u] = \text{GRU}(m_u(t), M_{t^-}[u])$$
- **Embedding & Link Prediction**: GNN message-passing computes temporal node embeddings $z_u(t), z_v(t)$, followed by a link classification decoder:
  $$\hat{Y}_{u,v,t} = \sigma(\text{MLP}_{dec}(z_u(t) \,\|\, z_v(t)))$$
- **Training Protocol**: Trained via binary cross-entropy on historical sequences with early stopping on validation AP.

---

### 3.6 Non-Parametric Addressable Historical Retrieval
- **Episodic Snapshot Memory**: Maintains an archive of past graph snapshots $\mathcal{H} = \{G_1, G_2, \dots, G_{t-1}\}$.
- **Query & Key Representation**: For current snapshot $G_t$ and historical snapshot $G_\tau$, compute normalized degree/affinity feature vectors $q_t = \phi(G_t)$ and $k_\tau = \phi(G_\tau)$.
- **Similarity Retrieval**: Retrieve the most structurally aligned historical snapshot:
  $$\tau^* = \arg\max_{\tau \le t - \Delta_{buffer}} \cos(q_t, k_\tau)$$
- **Historical Structural Decoding**: Compute candidate link scores by combining current local connectivity with retrieved structural affinity $G_{\tau^*}[u, v]$.

---

### 3.7 EdgeBank Baseline Implementation
- **All-History Mode**: Maintains an unbounded hash table $\mathcal{E}_{hist} = \{(u, v) : (u, v) \text{ observed at any } \tau < t\}$.
- **Bounded-A Mode**: Explicitly stores edges observed during Regime A only ($\tau \in [1, T_A]$).
- **Scoring**: Assigns positive prediction score if $(u, v) \in \mathcal{E}_{hist}$, with secondary recency weighting if tie-breaking is required.

---

### 3.8 Strict Zero-Leakage Controls
- **Temporal Leakage**: Strict prohibition of future information ($G_\tau$ for $\tau \ge t$).
- **Target Leakage**: Positive links evaluated at step $t$ are excluded from input graph $G_t$ during embedding generation.
- **Candidate Parity**: Exact identity of evaluated positive and negative candidate sets across all models.
- **Label Parity**: Exact identity of binary ground-truth labels across all compared methods.

---

### 3.9 Memory Capacity Response Surface Analysis
- **Grid Evaluation**: 5 recurrent dimensions $d_m \in \{16, 32, 64, 128, 256\} \times 4$ distractor durations $T_B \in \{10, 50, 100, 200\}$.
- **Response Surface Regression**:
  $$\text{AP} = \beta_0 + \beta_1 \log(d_m) + \beta_2 T_B + \beta_3 (\log(d_m) \cdot T_B) + \epsilon$$
- Quantifies whether expanding state capacity compensates for distractor duration decay.

---

### 3.10 Diagnostic Mechanism Interventions
- **TGN-NoMemory**: Evaluates TGN with persistent memory disabled ($M_t = \mathbf{0}$).
- **Memory Reset**: Resets node memory to zero at regime transitions ($M_{T_A + T_B} \leftarrow \mathbf{0}$).
- **Memory Shuffle**: Randomly permutes memory vectors across nodes at test onset to test the effect of corrupted structural associations.

---

### 3.11 Real-World Recurrence Episodes on SNAP CollegeMsg
- **Dataset**: SNAP CollegeMsg (1,899 nodes, 59,835 timestamped messages over 193 days).
- **Recurrence Episode Discovery**: Sliding 14-day window cross-correlation of community interaction matrices identifies $n = 4$ independent recurring episodes ($W_i \to W_j \to W_k$ where affinity $\text{Corr}(W_i, W_k) > 0.65$ while $\text{Corr}(W_i, W_j) < 0.25$).
- **Statistical Unit**: Each discrete recurrence episode constitutes an independent statistical observation ($n = 4$).
- **Bootstrap Protocol**: Non-parametric block bootstrap resampling across episodes to compute empirical 95% confidence intervals for $\Delta \text{AP}$.

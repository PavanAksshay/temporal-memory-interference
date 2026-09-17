# Minimal Mathematical Formulation

This document defines the minimal, rigorous mathematical formalism used in the paper, contrasting continuous autoregressive state compression with addressable historical memory.

---

## 1. System Formalism

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T})$ be a continuous-time temporal graph with static node set $\mathcal{V} = \{1, \dots, N\}$. 

At any discrete temporal observation step $t \in \{1, \dots, T\}$, the network is observed as a snapshot graph $G_t = (\mathcal{V}, \mathcal{E}_t)$, where $\mathcal{E}_t \subseteq \mathcal{V} \times \mathcal{V}$ denotes the set of edges active at time $t$.

The sequence of underlying generation regimes is governed by a discrete regime indicator $R_t \in \{\mathcal{A}, \mathcal{B}\}$, where each regime defines a latent community partition $C_R: \mathcal{V} \to \{1, \dots, K\}$ and an affinity matrix $P_R \in [0, 1]^{K \times K}$. 

Edge occurrence between nodes $u$ and $v$ at time $t$ is drawn independently as:
$$Y_{u,v,t} \sim \text{Bernoulli}\left( P_{R_t}[C_{R_t}(u), C_{R_t}(v)] \right)$$

---

## 2. Continuous Recurrent State Compression

A continuous recurrent temporal graph neural network (e.g., TGN) models the dynamic history of each node $u \in \mathcal{V}$ via a fixed-dimensional state vector $M_t[u] \in \mathbb{R}^{d_m}$.

Upon observing new events at time $t$, the state updates autoregressively:
$$M_t[u] = f_\theta\left( M_{t-1}[u], \mathcal{N}_t(u) \right)$$
where $\mathcal{N}_t(u)$ denotes the local interaction neighborhood of node $u$ at time $t$, and $f_\theta$ is a parameterized recurrent update function (such as a GRU cell).

The future link probability $\hat{Y}_{u,v,t+1}$ is predicted by decoding the joint state:
$$\hat{Y}_{u,v,t+1} = g_\phi\left( M_t[u], M_t[v], G_t \right)$$

### The Compression Limitation under Recurrence
Under the recurring regime sequence:
$$R_1, \dots, R_{T_A} = \mathcal{A} \quad \xrightarrow{\quad} \quad R_{T_A+1}, \dots, R_{T_A+T_B} = \mathcal{B} \quad \xrightarrow{\quad} \quad R_{T_A+T_B+1}, \dots = \mathcal{A}$$

The recurrent node state at the onset of recurrence is the result of $T_B$ successive lossy updates under regime $\mathcal{B}$:
$$M_{T_A + T_B}[u] = f_\theta^{(T_B)}\left( M_{T_A}[u], \{G_\tau\}_{\tau=T_A+1}^{T_A+T_B} \right)$$

Because $d_m \ll |\mathcal{V}|$ and $f_\theta$ is trained to predict immediate transitions, successive updates under orthogonal regime $\mathcal{B}$ overwrite coordinates in $M_t$ that encode regime $\mathcal{A}$ community structure, resulting in temporal memory interference.

---

## 3. Addressable Historical Memory

In contrast to compressing history into a single evolving state vector $M_t$, an addressable historical memory system preserves past observations as an indexable set:
$$\mathcal{M}_{hist} = \left\{ (\tau, G_\tau) : \tau < t \right\}$$

Given current query information $q_t = \phi(G_t)$, the system retrieves relevant historical context via an explicit addressing function:
$$H_t = \text{Retrieve}\left( q_t, \mathcal{M}_{hist} \right) = \sum_{\tau < t} \alpha(q_t, k_\tau) G_\tau$$
where $k_\tau = \phi(G_\tau)$ is the historical key representation and $\alpha(q_t, k_\tau)$ is a similarity-based attention weight (in our non-parametric diagnostic setting, top-1 cosine similarity $\tau^* = \arg\max_\tau \cos(q_t, k_\tau)$).

Future link prediction is then computed over the retrieved historical structure:
$$\hat{Y}_{u,v,t+1} = h_\psi\left( G_t[u, v], H_t[u, v] \right)$$

---

## 4. Analytical Distinction: Compression vs. Indexing

The fundamental distinction formalized in this paper is structural:

$$\begin{aligned}
\text{\textbf{Continuous Recurrent Model:}} \quad & \mathcal{H}_{<t} \xrightarrow{\quad \text{compress} \quad} M_t \in \mathbb{R}^{N \times d_m} \xrightarrow{\quad \text{predict} \quad} \hat{Y}_{t+1} \\
\text{\textbf{Addressable Historical Memory:}} \quad & \mathcal{H}_{<t} \xrightarrow{\quad \text{index} \quad} \{G_\tau\}_{\tau=1}^{t-1} \xrightarrow{\quad \text{query}(G_t) \quad} G_{\tau^*} \xrightarrow{\quad \text{predict} \quad} \hat{Y}_{t+1}
\end{aligned}$$

- **Continuous compression** forces all historical dynamics through a fixed bottleneck $M_t$, subjecting earlier dynamics to state overwriting by intervening regimes.
- **Addressable memory** decouples storage capacity from temporal sequence length, allowing direct random-access retrieval to historically relevant regimes regardless of intervening distractor duration $T_B$.

> [!NOTE]
> This formulation avoids asserting an impossibility theorem for all recurrent systems, providing instead a clear conceptual and mathematical contrast between online lossy state compression and addressable episodic retrieval.

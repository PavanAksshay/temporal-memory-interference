# Causal Experimental Logic and Information Hierarchy

## 1. Causal Experimental Sequence

The paper's experimental design is structured as a controlled causal perturbation protocol:

```
[ REGIME A ]
- Latent community structure generates predictive edge dynamics
- Historical oracle signal formed: S_A ~ Ber(P_A)
- Initial state M_A learned / compressed by continuous models
       ↓
[ REGIME B ] (Conflicting / Distractor Dynamics)
- Conflicting community structure generates orthogonal dynamics: S_B ~ Ber(P_B)
- Recurrent state updated across T_B time steps: M_{t} = f(M_{t-1}, G_t)
- Duration T_B systematically varied: T_B ∈ {25, 50, 100, 200}
       ↓
[ REGIME A RETURNS ] (Recurrence Phase)
- Original regime dynamics S_A reactivated
- Current snapshot G_t contains imperfect signal (partial edge drop / noise)
- Task: Predict future links Y_{t+1} in recurring Regime A
       ↓
[ MEASURE RECOVERABILITY ]
- Evaluate how much historical Regime A predictive information survives
- Compare continuous recurrent state vs. addressable historical memory vs. exact edge lookup
```

---

## 2. Baseline Diagnostic Roles

Each method in the experimental design serves a distinct diagnostic function to isolate the causal mechanism:

| Baseline | Operational Mechanism | Diagnostic Role |
|---|---|---|
| **Current-Only** | Evaluates link prediction using solely the current graph snapshot $G_t$ without historical state. | Establishes the **predictive baseline of immediate structure** ($AP \approx 0.7635$). Any method scoring below Current-Only suffers from negative historical interference. |
| **Continuous TGN** | Maintains a continuously updated GRU node memory bank $M_t \in \mathbb{R}^{N \times d_m}$ tracking all events through Regimes A, B, and recurring A. | Diagnoses whether **continuous autoregressive state compression** retains historically useful representations through conflicting temporal regimes. |
| **TGN-NoMemory** | Computes GNN node embeddings directly from local temporal neighborhoods without persistent node memory. | Tests whether failure in Continuous TGN is caused by **persistent state corruption** rather than static GNN message-passing architecture. |
| **EdgeBank (All-History / Bounded)** | Non-parametric hash-table storing exact historical edge occurrences $(u, v) \in \mathcal{E}_{hist}$. | Disentangles **exact historical pair repetition** from higher-order structural/community recurrence. |
| **Historical Retrieval** | Retrieves the most structurally similar historical graph snapshot $G_\tau$ ($\tau < t$) via cosine similarity and aggregates historical structure. | Evaluates whether **explicit, addressable non-parametric memory** preserves and recovers structural information that continuous state compresses away. |
| **Random Retrieval** | Retrieves an arbitrary, non-correlated historical snapshot $G_{\tau_{rand}}$ from the historical archive. | Serves as a **negative historical control** to verify that retrieval gains stem from regime-specific similarity rather than simply having an additional snapshot. |
| **Historical Oracle** | Accesses the true uncorrupted historical affinity matrix $S_A$ defined by the generative process. | Provides the **benchmark oracle under the specified generative definition** ($AP \approx 0.7904$), defining the total recoverable historical signal. |

---

## 3. Method Information Hierarchy

The following table makes explicit the exact information accessibility of each method, preventing conflation between memorization, compression, and retrieval:

| Method | Current Graph $G_t$ | Continuous Recurrent State $M_t$ | Exact Historical Pair Lookup $(u, v) \in \mathcal{E}_{hist}$ | Addressable Historical Snapshots $\{G_\tau\}$ | Historical Oracle Structure $S_A$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Current-Only** | **YES** | NO | NO | NO | NO |
| **Continuous TGN** | **YES** | **YES** (lossy update) | NO | NO | NO |
| **TGN-NoMemory** | **YES** | NO | NO | NO | NO |
| **EdgeBank (All-History)** | NO | NO | **YES** (unbounded hash) | NO | NO |
| **EdgeBank (Bounded A)** | NO | NO | **YES** (regime A hash) | NO | NO |
| **Historical Retrieval** | **YES** | NO | Optional | **YES** (indexed lookup) | NO |
| **Random Retrieval** | **YES** | NO | Optional | **YES** (random lookup) | NO |
| **Historical Oracle** | **YES** | NO | Optional | Optional | **YES** (ground-truth) |

> [!IMPORTANT]
> **Definitional Precision**: The *Historical Oracle* is NOT a theoretical upper bound for arbitrary graph learning models. It is the *benchmark oracle under the specified historical information definition* (evaluating links under the true latent community matrix $S_A$).

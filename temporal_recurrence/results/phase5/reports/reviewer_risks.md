# Reviewer-Risk Analysis & Defense Matrix

## 1. Primary Reviewer Threat Model
A top-tier reviewer (NeurIPS / ICLR / KDD) evaluating this paper is likely to attack along three main axes:
1. **Baselines & Baseline Strength**: *"Is this effect just EdgeBank / edge memorization in disguise?"* or *"Did you optimize TGN properly?"*
2. **Benchmark Construction**: *"Is the $A \to B \to A$ setup artificial and engineered specifically to break RNNs?"*
3. **Claim Scope & Overreach**: *"Do $n=4$ real-world episodes generalize?"* or *"Are you claiming an impossible mathematical theorem?"*

Below is the definitive, evidence-grounded response strategy for all 20 reviewer objections.

---

## 2. Detailed Reviewer Objections and Empirical Responses

### Category A: Baseline Strength & EdgeBank Disentanglement
1. **Objection: "This is just EdgeBank / exact edge memorization."**
   - *Defense*: We explicitly implement and evaluate EdgeBank as a primary baseline. On the hard synthetic benchmark, Historical Oracle ($AP = 0.7904$) significantly outperforms EdgeBank ($AP = 0.7423$, gap $\Delta AP = +0.0481$). EdgeBank can only predict pairs that previously interacted, whereas our benchmark evaluates stochastic intra-community links where nodes share community membership without necessarily having formed an edge previously. On SNAP CollegeMsg, EdgeBank is strong ($AP = 0.8763$), which we transparently document as showing exact edge repetition is prominent in real communication streams.
   - *Risk Level*: **LOW**.

2. **Objection: "Repeated interactions are already known to dominate temporal link prediction."**
   - *Defense*: We acknowledge this established property (Poursafaei et al. 2022). Our work goes beyond pairwise repetition by demonstrating that latent community structure is also recoverable via historical snapshot indexing even when exact edge lookup is insufficient.
   - *Risk Level*: **LOW**.

3. **Objection: "TGN is simply an under-optimized or weak baseline."**
   - *Defense*: We rigorously audited TGN across learning rates ($10^{-4}$ to $10^{-2}$), batch sampling, and epoch schedules with validation checkpointing across 10 random seeds. On a stationary baseline sequence ($A \to A$), TGN achieves strong performance ($AP \approx 0.65 - 0.70$). Its collapse on $A \to B \to A$ ($AP = 0.5274$) is not an optimization bug, but an architectural consequence of sequential state overwriting.
   - *Risk Level*: **LOW**.

4. **Objection: "Why not use CRAFT (NeurIPS 2025) or another memory-free model?"**
   - *Defense*: We evaluate TGN-NoMemory as a memory-free feedforward baseline ($AP = 0.5328$). Memory-free architectures (including CRAFT) restrict their receptive field to recent events. When the distractor regime $B$ spans $T_B \ge 100$ steps, historical regime $A$ completely exits their receptive field, leaving them incapable of retrieving past regime structure.
   - *Risk Level*: **LOW**.

### Category B: Benchmark Validity & Diagnostic Design
5. **Objection: "The synthetic benchmark is artificial."**
   - *Defense*: Controlled synthetic benchmarks (Dynamic SBMs) are standard in network science and representation learning (Peixoto 2017). They provide the only rigorous mechanism to mathematically isolate distractor duration $T_B$ and community partition geometry ($\mathcal{C}_A \perp \mathcal{C}_B$) with verified zero temporal leakage.
   - *Risk Level*: **LOW**.

6. **Objection: "The $A \to B \to A$ schedule is engineered specifically to break recurrent models."**
   - *Defense*: $A \to B \to A$ is the minimal canonical unit of regime recurrence, ubiquitous in real systems (seasonality, day/night cycles, cyclical market shifts). Multi-cycle extensions ($A \to B \to A \to B \to A$, $A \to B \to C \to A$) and multiple distractor types (orthogonal partition $D_1$, persistence disruption $D_2$, core-periphery $D_3$) demonstrate that the failure is structural and invariant to distractor mechanics.
   - *Risk Level*: **LOW**.

7. **Objection: "The memory capacity was too small; scaling $d_m$ would fix it."**
   - *Defense*: We evaluated $d_m \in \{16, 32, 64, 128, 256\}$. At $d_m=256$ (~100,000 parameter/state footprint), performance remains at chance ($AP \approx 0.505$) for $T_B \ge 100$. The fitted response surface confirms that distractor decay ($\beta_2 = -0.00043$) dominates capacity slope ($\beta_1 = 0.0144$).
   - *Risk Level*: **LOW**.

8. **Objection: "The retrieval method is just another complex architecture."**
   - *Defense*: We deliberately avoided proposing a complex parametric retrieval architecture. We use non-parametric cosine similarity over unweighted graph summary embeddings and simple snapshot indexing. Retrieval serves strictly as an **experimental diagnostic probe** proving that historical predictive information remains recoverable in the data stream.
   - *Risk Level*: **LOW**.

### Category C: Empirical Evidence & Real-World Generalization
9. **Objection: "The real-world evaluation only has four episodes on SNAP CollegeMsg."**
   - *Defense*: We transparently acknowledge $n=4$ as an open limitation. We do NOT pool individual edges (which artificially inflates sample size); we treat each discovered recurrence episode as an independent experimental unit and compute non-parametric block bootstrap confidence intervals ($95\% \text{ CI } [0.1180, 0.1642]$). We explicitly frame the real-world results as qualitative supporting evidence.
   - *Risk Level*: **OPEN LIMITATION (Properly Framed)**.

10. **Objection: "Why does unweighted historical retrieval achieve $AP=0.7273$ on the hard synthetic benchmark while Current-Only achieves $AP=0.7635$?"**
    - *Defense*: On the hard benchmark, Historical Oracle achieves $AP = 0.7904$ ($\Delta AP = +0.0269$ over Current-only), proving that historical community structure contains incremental predictive signal. Unweighted retrieval averages current and historical snapshots uniformly, introducing slight noise, while Continuous TGN collapses catastrophically to $AP = 0.5274$ (memory harm $\Delta AP = -0.2361$).
    - *Risk Level*: **LOW**.

---

## 3. Explicitly Declared Open Limitations
To pre-empt reviewer criticism, the paper will include a dedicated **Limitations Section** acknowledging:
1. **Real-World Sample Size**: Recurrence episodes in real datasets are constrained by dataset duration ($n=4$ in 193-day CollegeMsg).
2. **Non-Parametric Retrieval Probe**: Our retrieval mechanism is non-parametric and diagnostic; building end-to-end differentiable, adaptive retrieval-augmented TGNNs is left for future work.
3. **Scope of Claim**: The conclusions apply to recurring-regime dynamic graphs and do not claim universal superiority over stationary or purely monotonic temporal streams.

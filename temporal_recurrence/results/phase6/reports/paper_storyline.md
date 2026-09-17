# The Paper in One Story

This document provides a one-page narrative synthesis connecting every experimental step into a coherent scientific trajectory.

---

### The Narrative Arc

**1. Temporal graphs evolve, and history matters.**
Dynamic graphs in communication, finance, biology, and social systems rarely evolve as static, stationary processes. They exhibit non-stationary transitions, cyclic patterns, and recurring interaction regimes where relationships established in the distant past suddenly become relevant again.

**2. Contemporary temporal GNNs compress history into continuous recurrent states.**
Dominant architectures like TGN, JODIE, and DyRep maintain fixed-dimensional node memory vectors updated autoregressively upon each observed event. This continuous compression is designed to encode local temporal context into a compact state vector.

**3. When a previous regime recurs, continuous state must carry historical structure through conflicting dynamics.**
Consider an $A \to B \to A$ recurrence scenario: nodes form community affiliations in Regime $A$, transition through a conflicting/distractor Regime $B$ for duration $T_B$, and subsequently return to Regime $A$. To predict future links in recurring Regime $A$, a continuous recurrent model must preserve Regime $A$ representations across all intervening Regime $B$ updates.

**4. Intervening conflicting dynamics produce systematic historical overwriting.**
On our calibrated Dynamic SBM benchmark across 10 independent seeds, Continuous TGN exhibits monotonic performance collapse as distractor duration $T_B$ increases: AP falls from $0.5942$ ($T_B = 25$) to $0.5028$ ($T_B = 200$, near-chance level). The model actively overwrites earlier structural representations with conflicting intermediate interactions.

**5. Increasing recurrent state dimension within tested limits does not prevent overwriting.**
Evaluating a response surface across $d_m \in \{16, 32, 64, 128, 256\} \times T_B \in [10, 200]$ reveals that distractor duration decay ($\beta_2 = -0.00043$) dominates memory capacity slope ($\beta_1 = 0.0144$). Simply expanding the hidden state vector does not protect historical information against sustained conflicting state transitions.

**6. Addressable historical memory preserves and recovers the lost predictive signal.**
By decoupling storage capacity from sequential state updates, non-parametric addressable historical retrieval maintains an uncorrupted archive of past graph states. It consistently achieves $AP \approx 0.7273$ across all distractor durations, overcoming the capacity and interference bottlenecks of continuous state compression.

**7. Exact edge memorization explains a large fraction of recurrence, but structural retrieval captures higher-order community dynamics.**
Comparing against EdgeBank ($AP \approx 0.7397$) demonstrates that exact historical pair lookup accounts for $\approx 65\%$ of synthetic historical oracle gain. However, in evolving community structures where new links form between previously affiliated nodes, structural snapshot retrieval captures latent affiliations beyond raw pairwise repetition.

**8. Real-world communication streams exhibit qualitative recurrence where addressable memory consistently outperforms continuous tracking.**
In SNAP CollegeMsg across $n = 4$ independently discovered recurrence episodes, addressable historical storage achieves positive gain over continuous TGN in all 4 episodes ($\Delta AP > 0$), with exact edge memory dominating real interaction streams ($AP = 0.8763$).

**9. The scientific conclusion is a diagnostic characterization of temporal memory interference.**
Rather than asserting an impossibility theorem or claiming universal superiority of retrieval, the paper delivers a rigorous empirical characterization: *continuous autoregressive state compression in dynamic graphs is vulnerable to historical overwriting under conflicting dynamics, and addressable historical indexing provides an essential architectural inductive bias for capturing recurring graph dynamics.*

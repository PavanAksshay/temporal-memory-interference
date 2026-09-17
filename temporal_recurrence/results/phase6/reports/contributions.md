# Final Paper Contribution Structure

The paper delivers four precisely defined, scientifically defensible contributions.

---

### Contribution 1: Controlled Dynamic SBM Recurrence Benchmark
We introduce a mathematically calibrated Dynamic Stochastic Block Model (DSBM) diagnostic framework ($N = 300, K = 3, \rho = 0.10, M = 4.0$) featuring independent orthogonal community partitions, verified zero temporal/target leakage, and controlled distractor durations ($T_B$). This benchmark isolates the recoverability of historical information under regime recurrence ($A \to B \to A$) from confounding factors such as node churn, static density shifts, and trivial edge repetition.

---

### Contribution 2: Empirical Characterization of Temporal Memory Interference
Across 10 independent seeds and varying distractor durations ($T_B \in [25, 200]$), we provide empirical evidence that continuously updated recurrent node states in temporal GNNs (TGN) undergo systematic historical overwriting ($AP$ collapses from $0.5942$ to $0.5028$). Through a capacity response surface analysis across $d_m \in [16, 256]$, we demonstrate that distractor decay ($\beta_2 = -0.00043$) dominates capacity scaling ($\beta_1 = 0.0144$), showing that increasing recurrent state dimension within standard limits cannot eliminate this degradation.

---

### Contribution 3: Disentangling Exact Edge Memorization from Structural Retrieval
We benchmark continuous models against non-parametric addressable historical memory, demonstrating that explicit snapshot indexing preserves predictive regime information across all distractor durations ($AP \approx 0.7273$). Furthermore, by comparing against exact-edge memorization (EdgeBank, $AP \approx 0.7397$), we quantitatively decompose the historical signal: exact pair lookup explains $\approx 65\%$ of synthetic historical gain, while structural snapshot retrieval captures latent community affinities beyond exact pair recurrence.

---

### Contribution 4: Real-World Recurrence Evaluation on SNAP CollegeMsg
We evaluate the empirical analogue of regime recurrence in real-world social interaction streams using SNAP CollegeMsg across $n = 4$ discovered recurrence episodes. Non-parametric block bootstrapping confirms that addressable historical storage consistently recovers predictive performance over continuous recurrent tracking ($\Delta AP > 0$ across all 4 episodes), while revealing that exact edge repetition accounts for the dominant share of real-world historical signal ($AP = 0.8763$).

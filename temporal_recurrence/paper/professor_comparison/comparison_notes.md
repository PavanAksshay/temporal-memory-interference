# Comparative Review & Discussion Notes for Faculty Review

**Manuscript**: *When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks*  
**Evaluation Target**: Professor-Review / Comparison Version (`paper/professor_comparison/`) vs. Final Submission Draft (`paper/final/`)

---

## A. Structural and Narrative Differences from Publication Manuscript

| Dimension | Publication Submission Draft (`paper/final/`) | Professor Review Draft (`paper/professor_comparison/`) | Rationale for Professor Draft |
| :--- | :--- | :--- | :--- |
| **Pacing & Accessibility** | Highly condensed for strict conference page limits (8–9 pages body). Assumes deep familiarity with TGNN memory mechanics. | Pedagogical first 2–3 pages with intuitive explanations of continuous stream compression, dynamic graphs, and regime recurrence before formal math. | Allows advisors and cross-area faculty to assess the core conceptual motivation without parsing dense notation upfront. |
| **Hypothesis Formalization** | Implicitly integrated across experimental narratives and contribution bullets. | Explicitly codified as four testable hypotheses (**H1–H4**) in Section 1.3, referenced directly in Results sections. | Enhances scientific rigor and facilitates direct hypothesis-testing evaluation. |
| **Synthetic Benchmark Framing** | Benchmark details condensed into methods and setup sections. | Dedicated Section 4 detailing the Dynamic Stochastic Block Model (DSBM), persistence parameters ($\lambda_A, \lambda_B$), and density matching ($\rho=0.10$). | Highlights experimental control and eliminates suspicions of trivial density artifacts. |
| **Table & Figure Suite** | Modular inclusion of primary figures and tables across separate subsection files. | Unified 10-figure and 10-table presentation directly integrated into a comprehensive manuscript and appendix. | Provides complete self-contained context for review without jumping between supplementary files. |

---

## B. Sections Restructured and Clarified for Review

1. **Introduction (Section 1)**:
   - Restructured into four clear subsections: *1.1 Problem (Dynamic Relational Streams)*, *1.2 Motivation (Temporal Recurrence & Benchmark Gap)*, *1.3 Central Research Question & Hypotheses (H1–H4)*, and *1.4 Summary of Contributions*.
   - Clearly distinguishes between *standard dynamic link prediction* and *in-stream historical recoverability*.

2. **Related Work & Positioning (Section 2 & Table~\ref{tab:literature_positioning})**:
   - Explicitly positions the work across six distinct literature categories: Continuous-Time Recurrent TGNNs, Long-History & Memory-Free Models, Exact Edge Tables, Continual Graph Learning, Episodic Graph Memory, and Periodic Models.
   - Includes 30 verified bibliographic citations from verified peer-reviewed venues (NeurIPS, ICLR, KDD, AAAI, CIKM, TKDE).

3. **Controlled Synthetic Benchmark (Section 4)**:
   - Details why naive unconstrained benchmarks fail (density shift confounds) and presents the mathematical construction of the first-order Markov persistence process with matched marginal density ($\rho = 0.10$).
   - Explicitly walks through the 8-point temporal non-anticipation and causality audit (Appendix~\ref{app:leakage}).

---

## C. Deliberately Conservative Scientific Framing

To maintain strict scientific integrity, several claims were deliberately hardened and scoped to empirical boundaries:

1. **No "Mathematical Proof of Inevitable Forgetting"**:
   - *Draft Framing*: Temporal memory interference is presented as an **empirical characterization** of continuous recurrent compression under conflicting dynamics, rather than an unproven theoretical impossibility theorem.
2. **MA-TGN as an Empirical Diagnostic Intervention**:
   - *Draft Framing*: MA-TGN is explicitly introduced as a diagnostic architecture designed to test whether addressable historical states preserve accessible structural information when overwritten in recurrent GRUs. It is **not** pitched as a universally dominant replacement for exact edge lookup tables.
3. **Transparent Recognition of EdgeBank Dominance**:
   - *Draft Framing*: Acknowledges that on real-world networks with heavy pairwise repetition (SNAP CollegeMsg $\text{AP} = 0.8763$, SNAP Bitcoin-OTC $\text{AP} = 0.7753$), exact-edge memorization (EdgeBank) substantially outperforms all neural models. MA-TGN's advantage is scoped to *structural community recurrence* where edge overlap is suppressed.
4. **Addressing Mechanism Neutrality**:
   - *Draft Framing*: Honestly reports that while learned attention enables dynamic routing during regime transitions, deterministic cosine key similarity achieves comparable link prediction accuracy ($0.6519$ vs. $0.6518$) on static community partitions.
5. **Exact Real-World Sample Size Reporting**:
   - *Draft Framing*: Clarifies that real-world validations operate over $n=4$ natural recurrence episodes per dataset, avoiding inflated claims of universal multi-domain generalization.

---

## D. Key Figures and Tables for Discussion with Professor

1. **Figure 1 & Table 1 (Conceptual Benchmark & Parameters)**:
   - *Discussion Point*: How the $\mathcal{A} \to \mathcal{B} \to \mathcal{A}$ setup cleanly isolates memory overwriting from trivial recency or density cues.
2. **Figure 2 & Table 2 (Historical Recoverability vs. Distractor Duration $T_B$)**:
   - *Discussion Point*: The degradation of unguided historical sampling (Random Retrieval: $0.7423 \to 0.7193$) and the gap between explicit structural heuristics ($0.760$) and unfeatured neural message passing ($0.652$).
3. **Figure 3 & Table 4 (Recurrent Memory Capacity Scaling $d_m \in [16, 256]$)**:
   - *Discussion Point*: Why expanding recurrent state capacity ($+412\%$ parameters) provides negligible protection ($<0.008$ AP gain) against long conflicting distractor regimes.
4. **Figure 6 & Table 5 (Exact vs. Structural Recurrence Decomposition)**:
   - *Discussion Point*: The double dissociation between EdgeBank (dominates Condition A at $0.8884$, collapses on Condition B at $0.5774$) and Historical Retrieval ($+0.0351$ AP advantage on Condition B, $p < 10^{-6}$).
5. **Table 7 & Figure 9 (Real-World Recurrence Episodes in CollegeMsg and Bitcoin-OTC)**:
   - *Discussion Point*: The consistency of episodic recovery across natural recurrence episodes and the practical tradeoff with exact-edge caching.
6. **Table 8 & Figure 10 (Memory Footprint and Per-Candidate Latency Profiling)**:
   - *Discussion Point*: Exact analytical accounting of $827.5\text{ KB}$ RAM for $K=10$ checkpoints on $300$ nodes, with sub-$5\mu\text{s}$ per-candidate scoring latency.

---

## E. Core Scientific Questions for Faculty Evaluation

1. **Framing & Terminology**: Is the terminology *Temporal Memory Interference* and *Historical Recoverability* sufficiently clear and well-delineated from continual graph learning / catastrophic forgetting?
2. **Experimental Controls**: Does the combination of density matching ($\rho=0.10$), unfeatured node IDs, and 8-point temporal non-anticipation satisfy the highest standards of causality and fairness?
3. **Decomposition Impact**: Is the distinction between *exact pairwise edge memorization* (EdgeBank) and *latent structural community recurrence* presented as the central conceptual takeaway?
4. **Boundary Scoping**: Are the acknowledged limitations (synthetic SBM assumptions, $n=4$ real-world episodes, diagnostic probe nature) appropriately conservative without undermining the empirical contributions?

---

## F. Objective Dimensional Assessment (Non-Numerical)

| Assessment Dimension | Implementation Status & Evidence in Professor Draft |
| :--- | :--- |
| **Clarity & Readability** | Intuitive opening narrative, structured progression from problem formulation to empirical decomposition. |
| **Scientific Motivation** | Real-world recurrence examples (academic, financial, social) paired with clear gap analysis in standard benchmarks. |
| **Novelty Positioning** | Clear separation from CRAFT, DyGFormer, EdgeBank, and task-continual graph learning; supported by Table~\ref{tab:literature_positioning}. |
| **Experimental Control** | Strictly matched marginal edge density ($\rho=0.10$), independent partition seeds, 8-point non-anticipation audit. |
| **Baseline Completeness** | Evaluates 8 distinct baseline profiles: Historical Oracle, Current-Only, EdgeBank (Bounded \& All-History), Hist. Retrieval Probe, Random Retrieval, Continuous TGN, TGN-NoMemory, MA-TGN. |
| **Statistical Strength** | Canonical 10-seed evaluations with mean $\pm$ standard deviation reporting; paired $t$-tests for condition comparisons. |
| **Real-World Validation** | Natural recurrence episodes across SNAP CollegeMsg and Bitcoin-OTC ($n=4$ episodes each) with episode-by-episode reporting. |
| **Computational Accounting** | Analytical memory derivations ($827.5\text{ KB}$ for $K=10$) and explicit per-candidate latency measurements ($0.91$--$4.79\mu\text{s}$). |
| **Limitations & Scope** | Fully transparent discussion of synthetic assumptions, sample size boundaries, and EdgeBank's dominance on exact-edge graphs. |

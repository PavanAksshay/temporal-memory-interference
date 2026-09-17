# Phase 5 Verdict: Literature Grounding, Novelty Decision & Publication Readiness

## 1. Final Classification
**VERDICT: B — NOVEL BUT MUST BE NARROWLY POSITIONED**

### Detailed Justification:
The literature search confirms that no prior work has formally parameterized the continuous-time distractor duration response curve ($T_B$) or constructed a controlled benchmark for recurring dynamic community regimes ($A \to B \to A$). 

However, because:
1. Poursafaei et al. (NeurIPS 2022) already established that exact edge memorization (EdgeBank) explains substantial dynamic link prediction performance in real networks, and
2. Yi et al. (NeurIPS 2025, CRAFT) already established that memory-free models are competitive for general future link prediction,

our work **must NOT be positioned** as a general claim that "memory is universally necessary" or as a "new SOTA retrieval architecture." 

Instead, the **surviving narrow contribution** is:
> **A rigorous empirical characterization of temporal memory interference in continuous recurrent temporal GNNs, demonstrating that lossy autoregressive state compression overwrites historically predictive regime information as distractor duration increases, and establishing that addressable non-parametric historical memory preserves and recovers this information across distractor durations.**

---

## 2. Final Literature-Grounded Central Claim
> *"Under controlled recurring-dynamics benchmarks, continuously updated recurrent temporal representations exhibit systematic historical overwriting: predictive information from a previously relevant regime becomes inaccessible after sufficiently long conflicting dynamics. Addressable historical retrieval preserves and recovers this information, outperforming continuous recurrent state models across varying distractor durations. In synthetic benchmarks with evolving community structure, structural historical retrieval provides predictive value beyond exact edge memorization (EdgeBank), and the phenomenon exhibits qualitative analogues in recurring interaction episodes in real-world temporal networks."*

---

## 3. Publication Readiness Matrix Summary

| Dimension | Audit Status | Key Evidence |
|---|---|---|
| **Empirical Robustness** | **FROZEN & VERIFIED** | Hard benchmark ($N=300$) replicates across 10 seeds ($AP_{oracle} = 0.7904 > AP_{current} = 0.7635$). |
| **Baseline Parity** | **VERIFIED** | EdgeBank, TGN, TGN-NoMemory, Retrieval, and Oracle evaluated on identical candidate sets. |
| **Statistical Validity** | **VERIFIED** | Episode-level units on CollegeMsg ($n=4$) with block bootstrap ($0 \notin CI(\Delta AP)$). |
| **Literature Novelty** | **NARROWED & GROUNDED** | Disentangled from EdgeBank (structural vs pairwise) and CRAFT (recurring vs novel edges). |
| **Reviewer Threat Defenses** | **PRE-EMPTED** | 20 objections systematically answered in reviewer attack matrix; open limitations declared. |
| **Terminology Governance** | **REFORMED** | Retired "recency bias" in favor of "temporal memory interference". |

---

## 4. Next Step
The project is fully frozen, audited, and literature-grounded. The next phase will be drafting the manuscript according to the established Balanced Positioning and Recommended Title:
> **"When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks"**

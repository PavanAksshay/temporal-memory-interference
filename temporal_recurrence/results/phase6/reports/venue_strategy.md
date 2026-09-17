# Venue Strategy and Reviewer Risk Analysis

This document evaluates suitable publication venue tiers for the manuscript based on its classification as a **Hybrid Diagnostic Benchmark + Empirical Phenomenon Paper**.

---

## 1. Venue Tier Categorization & Fit Analysis

### Tier 1: Premier Machine Learning & Data Mining Conferences
**Venues**: NeurIPS (Datasets and Benchmarks Track / Main Track), ICML, ICLR, KDD, The Web Conference (WWW).

- **Fit Rating**: **HIGH** (Specifically NeurIPS Datasets & Benchmarks Track or ICLR/KDD).
- **Why It Fits**:
  - NeurIPS Datasets & Benchmarks Track explicitly values rigorous diagnostic frameworks, anti-leakage audits, reproducibility checklists, and empirical characterizations of foundational GNN limitations.
  - ICLR and KDD value deep empirical investigations into representation learning dynamics in structured and temporal domains.
- **Main Reviewer Risk**:
  - Reviewers demanding a complex, new learned neural retrieval architecture rather than accepting a diagnostic non-parametric probe.
  - Reviewers claiming EdgeBank or CRAFT already solved memory limitations.
- **Preemptive Defenses**:
  - Clearly frame the paper as an empirical phenomenon + diagnostic benchmark paper.
  - Feature the explicit quantitative decomposition against EdgeBank (~65% exact pair lookup vs. ~35% structural community retrieval).

---

### Tier 2: Established AI & Graph / Data Mining Conferences
**Venues**: AAAI, IJCAI, WSDM, SDM, ECML-PKDD, LOG (Learning on Graphs Conference).

- **Fit Rating**: **VERY HIGH**.
- **Why It Fits**:
  - The Learning on Graphs (LOG) conference and WSDM/SDM are highly receptive to foundational diagnostic studies on graph neural network mechanics and rigorous benchmarking.
  - AAAI/IJCAI have strong tracks in machine learning fundamentals and benchmark auditing.
- **Main Reviewer Risk**:
  - Reviewers questioning real-world dataset diversity (single real-world dataset SNAP CollegeMsg with $n=4$ episodes).
- **Preemptive Defenses**:
  - Emphasize that the synthetic benchmark is the primary controlled experimental instrument, with CollegeMsg serving as transparent qualitative corroboration.

---

### Tier 3: Specialized Journals & Top Workshops
**Venues**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), IEEE TKDE, JMLR, or NeurIPS/ICML Graph Representation Learning Workshops.

- **Fit Rating**: **MODERATE / CONTINGENCY**.
- **Why It Fits**:
  - High page limits in journals allow complete verbatim inclusion of all 10-seed response surfaces, leakage audits, and extensive appendix proofs.

---

## 2. Recommended Primary Submission Target

> **Primary Recommendation**: **NeurIPS (Datasets & Benchmarks Track)** or **Learning on Graphs (LOG)**.
> 
> **Rationale**: These venues have reviewers who specifically reward rigorous experimental protocols, ablation audits, and zero-leakage diagnostic benchmarks over superficial incremental neural architectures.

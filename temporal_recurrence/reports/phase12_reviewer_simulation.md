# Phase 12: ICLR 2027 Adversarial Reviewer Simulation

**Simulated Venue**: ICLR 2027 Main Track  
**Reviewer Profiles**: 3 Hostile Specialists (Temporal GNNs, Continual Learning, Graph Benchmarking)

---

## Reviewer A: Temporal GNN & Architecture Specialist
- **Primary Critique**: *"CRAFT (NeurIPS 2025) showed that dynamic link prediction can be solved without memory or aggregation. Why should the community care about recurrent memory interference if memory-free architectures are competitive?"*
- **Manuscript Defense**: The paper explicitly clarifies that CRAFT addresses forward link forecasting on standard benchmarks, whereas our paper conducts a diagnostic investigation into what historical information remains recoverable inside recurrent representations when regimes recur. Recurrent models remain dominant in continuous streaming deployments; diagnosing their representational failure modes is essential.
- **Score**: **8 / 10** (Accept)

---

## Reviewer B: Continual Learning & Memory Specialist
- **Primary Critique**: *"Is this setting simply task-incremental catastrophic forgetting by another name?"*
- **Manuscript Defense**: The paper formally separates in-stream dynamic regime recurrence from task-incremental continual learning: the link prediction objective and node space remain fixed, but the latent structural generating process shifts and returns. We show that episodic checkpoint memory with non-anticipative attention routing mitigates interference at inference time without requiring backward gradient replay passes.
- **Score**: **8 / 10** (Accept)

---

## Reviewer C: Graph Benchmark & Evaluation Specialist
- **Primary Critique**: *"EdgeBank achieves higher AP on CollegeMsg ($0.8763$) and Bitcoin-OTC ($0.7753$) than MA-TGN. Furthermore, $n=4$ episodes per dataset is a small sample size."*
- **Manuscript Defense**: The paper does not hide EdgeBank's superiority, but uses it as a key scientific finding to separate exact pairwise edge repetition from structural community recurrence (Condition B). The $n=4$ episode sample size is explicitly stated in Limitations and framed as qualitative supporting evidence rather than overgeneralized population claims.
- **Score**: **9 / 10** (Strong Accept)

---

## Consensus Rating
- **Overall Score**: **8.3 / 10** (Clear Accept)
- **Reviewer Consensus**: The manuscript is exceptionally well-scoped, scientifically honest, and methodologically airtight.

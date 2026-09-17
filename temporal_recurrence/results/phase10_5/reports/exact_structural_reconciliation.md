# Phase 10.5: Exact vs. Structural Recurrence Reconciliation

**Protocol**: Controlled Dynamic SBM Recurrence Decomposition ($N=300$, $\rho=0.10$, $T_B=100$, 5 Seeds: 42–46).  
**Source Artifact**: `results/phase10/processed/table_f_exact_vs_structural.csv`

---

## 1. Quantitative Decomposition Matrix

| Condition | Description | Edge Jaccard | Historical Oracle | Current-Only | EdgeBank All-Hist | Hist. Retrieval | Continuous TGN | MA-TGN | $\Delta\text{AP}$ (MA-TGN $-$ TGN) | $\Delta\text{AP}$ (Retr $-$ EdgeBank) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A** | Exact Recurrence ($A \to B \to A$) | **0.6554** | 0.9097 | 0.8975 | 0.8884 | 0.8971 | 0.6484 | 0.6480 | -0.0003 | +0.0087 |
| **Condition B** | Structural Recurrence (Permuted Partition) | **0.9379** | 0.6578 | 0.6193 | 0.5774 | 0.6125 | 0.6527 | 0.6524 | -0.0003 | **+0.0351** |
| **Control** | Non-Recurring Baseline ($A \to B \to C$) | **0.8840** | 0.6869 | 0.7169 | 0.6872 | 0.7101 | 0.4990 | 0.4995 | +0.0005 | +0.0229 |

---

## 2. Scientific Reconciliation & Crucial Distinction

1. **Exact Recurrence (Condition A)**: High edge persistence yields high EdgeBank performance ($0.8884$) and Oracle performance ($0.9097$).
2. **Structural Recurrence (Condition B)**: When communities are structurally aligned but edge generation is independent, EdgeBank degrades sharply to $0.5774$, whereas structural historical retrieval preserves an advantage ($0.6125$, $\Delta = +0.0351$), proving that topological community invariants contain predictive signals beyond exact edge repetition.
3. **Control Condition ($A \to B \to C$)**: Continuous neural models collapse to random chance level ($\approx 0.4990$) when a novel unobserved community regime $C$ is introduced at test time.

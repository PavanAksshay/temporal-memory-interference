# Professor-Review / Comparison Manuscript Version

**Paper Title**: *When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks*

---

## 1. Purpose of this Version

This repository directory contains a **separate, professor-review / comparative draft** of the research paper. The purpose of this version is to provide a clean, pedagogical, and conventionally structured academic research paper draft that can be reviewed and compared directly against the publication-hardened submission manuscript (`paper/final/`).

### Scientific Integrity Invariants
- **Zero Result Modifications**: All empirical values, tables, and experimental metrics are frozen and match canonical evidence artifacts from Phase 10.5, Phase 11, and Phase 12.
- **Unmodified Final Manuscript**: The existing publication manuscript in `paper/final/` has remained completely untouched.
- **Zero Fabricated Evidence**: No experiments, numbers, datasets, citations, or claims were invented.

---

## 2. Directory Structure

```
paper/professor_comparison/
├── main.tex                                            # Comprehensive academic manuscript source
├── references.bib                                      # 30 verified bibliographic entries
├── figures/                                            # Publication-grade vector PDF figures (10 figures)
│   ├── fig1_benchmark_concept.pdf
│   ├── fig2_tb_response_curve.pdf
│   ├── fig3_capacity_scaling.pdf
│   ├── fig4_component_ablation.pdf
│   ├── fig5_memory_budget_vs_ap.pdf
│   ├── fig6_recurrence_decomposition.pdf
│   ├── fig7_reexposure_recovery.pdf
│   ├── fig8_matgn_attention_routing.pdf
│   ├── fig9_cross_domain_comparison.pdf
│   └── fig10_tradeoff_accuracy_cost.pdf
├── tables/                                             # Table components and assets
├── appendix/
│   └── supplementary_material.tex                      # Complete hyperparameter & audit supplementary material
├── temporal_graph_recurrence_professor_comparison.pdf # Compiled 18-page PDF document
├── comparison_notes.md                                 # In-depth side-by-side comparison notes for review
└── README.md                                           # Build instructions and manifest (this document)
```

---

## 3. Alternative Working Titles

While the canonical paper title is:
> **"When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks"**

The following two alternative titles are provided for prospective discussion:
1. *Temporal Memory Interference in Dynamic Graph Networks: Characterizing Historical Recoverability Under Recurring Dynamics*
2. *On the Limits of Recurrent State Compression in Dynamic Graph Neural Networks under Recurring Interaction Regimes*

---

## 4. How to Compile the Manuscript

### Compilation using Tectonic (Recommended, Self-Contained)
```bash
cd paper/professor_comparison/
tectonic main.tex --outdir .
cp main.pdf temporal_graph_recurrence_professor_comparison.pdf
```

### Compilation using Standard TeX Live / MacTeX
```bash
cd paper/professor_comparison/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cp main.pdf temporal_graph_recurrence_professor_comparison.pdf
```

---

## 5. Canonical Evidence Sources

All numerical values reported across the 10 main tables, figures, and text are traceable directly to primary frozen evidence files:
- `results/phase10_5/processed/evidence_registry.csv`
- `results/phase10_5/processed/claim_audit.csv`
- `results/phase10_5/processed/numerical_consistency_matrix.csv`
- `results/phase12/literature_matrix.csv`
- `results/phase12/novelty_matrix.csv`
- `results/phase12/claim_audit.csv`
- `results/phase12/reviewer_attack_matrix.csv`
- `results/phase10_5/reports/` (baseline, exact-structural, leakage, and memory reconciliation reports)

---

## 6. Key Structural Differences from Final Manuscript

1. **Pedagogical Exposition**: The introduction and problem formulation include structured pedagogical explanations of temporal graphs, continuous state compression, and regime recurrence before dense mathematical definitions.
2. **Explicit Hypotheses**: Hypotheses **H1** (Duration-dependent interference), **H2** (Capacity limits), **H3** (Addressable recovery), and **H4** (Exact vs. structural decomposition) are stated upfront in Section 1.3 and explicitly addressed across corresponding results subsections.
3. **Dedicated Synthetic Benchmark Section**: Section 4 provides a standalone breakdown of the $\mathcal{A} \to \mathcal{B} \to \mathcal{A}$ dynamic SBM generator, density-matching invariants, and leakage controls.
4. **Conservative & Delineated Claim Boundaries**: MA-TGN is explicitly framed as an empirical diagnostic intervention rather than a universal replacement for exact-match caching. EdgeBank's strength on exact-repetition datasets ($0.8763$ and $0.7753$) is highlighted as a fundamental boundary condition.
5. **Expanded 10-Table / 10-Figure Suite**: Incorporates the full suite of 10 figures and 10 tables directly within the body and appendix for thorough review.

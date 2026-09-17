# Final Submission Package: When History Recurs

**Paper Title**: *When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks*  
**Track**: Deep Learning on Dynamic Graphs / Representation Learning  
**Target Publication**: Top-tier Machine Learning / Graph Learning Conference

---

## Package Directory Contents

```
paper/final/
├── main.tex                             # Complete LaTeX manuscript
├── references.bib                       # BibTeX bibliography
├── temporal_graph_recurrence_manuscript.pdf # Compiled publication PDF
├── reproducibility_checklist.md         # Submission checklist
├── README.md                            # Package documentation
├── sections/                            # Modular LaTeX section sources
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── problem_formulation.tex
│   ├── experimental_setup.tex
│   ├── methods.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── limitations.tex
│   └── conclusion.tex
├── appendix/                            # Supplementary material
│   └── supplementary_material.tex
├── figures/                             # High-resolution publication figures (PDF/PNG)
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
└── tables/                              # Canonical processed CSV data tables
```

---

## Reproducibility Commands

To reproduce the complete experimental suite from scratch:
```bash
python3 experiments/phase10_scientific_strengthening.py
```

To compile the publication PDF:
```bash
python3 scripts/generate_single_pdf_manuscript.py
```

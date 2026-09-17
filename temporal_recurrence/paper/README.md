# When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks

This directory contains the complete publication-ready LaTeX source, figures, tables, appendix, and reproduction specifications for the manuscript.

---

## 1. Directory Structure

```
paper/
├── main.tex                            # Master LaTeX document
├── references.bib                      # BibTeX references
├── README.md                           # Overview and compilation instructions
├── reproducibility.md                  # Comprehensive reproduction guide
├── figures/                            # Publication vector figures (PDF and PNG)
│   ├── fig1_benchmark_concept.pdf
│   ├── fig2_tb_response_curve.pdf
│   ├── fig3_capacity_scaling.pdf
│   ├── fig4_recurrence_decomposition.pdf
│   ├── fig5_main_comparison_tb100.pdf
│   ├── fig6_collegemsg_episodes.pdf
│   ├── fig_app_convergence.png
│   └── fig_app_reexposure.png
├── tables/                             # LaTeX table definitions
├── appendix/
│   └── supplementary_material.tex      # Full appendices (A through J)
├── sections/                           # Modular manuscript sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── problem_formulation.tex
│   ├── methods.tex
│   ├── experimental_setup.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── limitations.tex
│   └── conclusion.tex
├── supplementary/                      # Supplementary data and configurations
└── build/
    └── manuscript.pdf                  # Rendered submission PDF
```

---

## 2. Compilation Instructions

### Using Tectonic (Self-Contained Engine)
```bash
tectonic paper/main.tex -o paper/build/
mv paper/build/main.pdf paper/build/manuscript.pdf
```

### Using Standard TeXLive / MacTeX
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
mv main.pdf build/manuscript.pdf
```

---

## 3. Environment & Hardware Specifications

- **Operating System**: macOS Darwin 24.6.0 (Apple Silicon ARM64)
- **Python**: 3.11.x
- **Deep Learning Frameworks**: PyTorch 2.4.0, PyG (torch-geometric) 2.6.0
- **Data & Scientific Libraries**: NumPy 1.26.4, SciPy 1.13.1, Pandas 2.2.2, Scikit-learn 1.5.1, Matplotlib 3.9.0
- **Random Evaluation Seeds**: 10 fixed seeds $[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]$
- **Independent SBM Partition Seeds**: $\pi_A: 101, \pi_B: 202, \pi_C: 303$

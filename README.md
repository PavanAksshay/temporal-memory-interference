# When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks

[![Paper (Final Manuscript)](https://img.shields.io/badge/Paper-Final%20Manuscript-blue.svg)](temporal_recurrence/paper/final/temporal_graph_recurrence_manuscript.pdf)
[![Paper (Professor Comparison Version)](https://img.shields.io/badge/Paper-Faculty%20Review%20Draft-green.svg)](temporal_recurrence/paper/professor_comparison/temporal_graph_recurrence_professor_comparison.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical and methodological investigation into **temporal memory interference** in continuous-time Dynamic Graph Neural Networks (TGNNs) under recurring structural interaction regimes.

---

## 📄 Manuscripts & Paper Versions

This repository contains two parallel versions of the research manuscript:

1. **[Final Publication Manuscript](temporal_recurrence/paper/final/)**:
   - PDF: [`temporal_graph_recurrence_manuscript.pdf`](temporal_recurrence/paper/final/temporal_graph_recurrence_manuscript.pdf)
   - Source LaTeX: [`main.tex`](temporal_recurrence/paper/final/main.tex)
   - Reproducibility Checklist: [`reproducibility_checklist.md`](temporal_recurrence/paper/final/reproducibility_checklist.md)

2. **[Professor-Review / Comparison Version](temporal_recurrence/paper/professor_comparison/)**:
   - PDF: [`temporal_graph_recurrence_professor_comparison.pdf`](temporal_recurrence/paper/professor_comparison/temporal_graph_recurrence_professor_comparison.pdf)
   - Source LaTeX: [`main.tex`](temporal_recurrence/paper/professor_comparison/main.tex)
   - Side-by-Side Review Guide: [`comparison_notes.md`](temporal_recurrence/paper/professor_comparison/comparison_notes.md)
   - Supplementary Appendix: [`appendix/supplementary_material.tex`](temporal_recurrence/paper/professor_comparison/appendix/supplementary_material.tex)

---

## 🔬 Key Scientific Findings

- **Duration-Dependent Recurrent Degradation**: Continuous recurrent node representations exhibit systematic degradation as intervening conflicting distractor duration ($T_B$) increases.
- **Capacity Limitations**: Quadrupling recurrent memory capacity ($d_m = 64 \to 256$, $+412\%$ parameters) yields $<0.008$ AP improvement at $T_B=100$, demonstrating that state scaling alone does not resolve sequential overwriting.
- **Exact vs. Structural Recurrence**: Exact edge memorization (EdgeBank) excels when identical pairwise edges recur ($\text{AP} = 0.8841$), but collapses under structural community recurrence ($\text{AP} = 0.5774$), where structural retrieval retains an empirical advantage ($+0.0351$ AP, $p < 10^{-6}$).
- **Addressable Memory Diagnostic (MA-TGN)**: Addressable episodic checkpoint storage consistently outperforms continuous TGN on multi-week real-world recurrence episodes in SNAP CollegeMsg ($\Delta\text{AP} = +0.0583$) and SNAP Bitcoin-OTC ($\Delta\text{AP} = +0.1243$).

---

## 📁 Repository Structure

```text
├── temporal_recurrence/
│   ├── paper/
│   │   ├── final/                  # Hardened camera-ready submission manuscript
│   │   └── professor_comparison/   # Pedagogical faculty-review comparison draft
│   ├── src/
│   │   ├── baselines/              # EdgeBank, Current-Only, Historical Oracle
│   │   ├── evaluation/             # Non-anticipation leak audit & metrics
│   │   ├── generator/              # Dynamic SBM benchmark generator
│   │   └── models/                 # Continuous TGN & Memory-Augmented TGN (MA-TGN)
│   ├── experiments/                # Parameterized sweep runners
│   ├── results/                    # Phase 0 through Phase 12 empirical evidence artifacts
│   ├── scripts/                    # Reproduction and figure generation scripts
│   ├── tests/                      # Unit and integration test suite
│   └── webapp/                     # Interactive Vite/React simulation & attention dashboard
├── paper -> temporal_recurrence/paper
├── results -> temporal_recurrence/results
└── README.md
```

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/PavanAksshay/temporal-memory-interference.git
cd temporal-memory-interference/temporal_recurrence
pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/
```

### Building the Papers
```bash
# Professor-Review Comparison Draft
cd temporal_recurrence/paper/professor_comparison
tectonic main.tex --outdir .

# Final Publication Manuscript
cd ../final
tectonic main.tex --outdir .
```

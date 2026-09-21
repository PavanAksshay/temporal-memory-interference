# When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks

[![Paper (Final Manuscript)](https://img.shields.io/badge/Paper-Final%20Manuscript-blue.svg)](temporal_recurrence/paper/final/temporal_graph_recurrence_manuscript.pdf)
[![Paper (Professor Comparison Version)](https://img.shields.io/badge/Paper-Faculty%20Review%20Draft-green.svg)](temporal_recurrence/paper/professor_comparison/temporal_graph_recurrence_professor_comparison.pdf)
[![Frontend Dashboard](https://img.shields.io/badge/Frontend-Vite%20%2B%20React%2019-61dafb.svg)](temporal_recurrence/webapp)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](requirements.txt)
[![Build Status](https://img.shields.io/badge/Build-Passing-success.svg)](temporal_recurrence/tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical and methodological investigation into **temporal memory interference** in continuous-time Dynamic Graph Neural Networks (TGNNs) under recurring structural interaction regimes ($\mathcal{A} \to \mathcal{B} \to \mathcal{A}$).

---

## 📑 Table of Contents
1. [Overview & Research Motivation](#-overview--research-motivation)
2. [Manuscripts & Review Drafts](#-manuscripts--review-drafts)
3. [Key Scientific Findings & Hypotheses](#-key-scientific-findings--hypotheses)
4. [Full Research Paper Tables Registry (14 Tables)](#-full-research-paper-tables-registry)
5. [Interactive Simulation Dashboard & Frontend](#-interactive-simulation-dashboard--frontend)
6. [Repository Architecture](#-repository-architecture)
7. [Installation & Quickstart](#-installation--quickstart)
8. [Reproducing Paper Experiments & Compiling Manuscripts](#-reproducing-paper-experiments--compiling-manuscripts)
9. [Citation](#-citation)

---

## 🔬 Overview & Research Motivation

Continuous-time Temporal Graph Neural Networks (TGNNs) continually compress sequential interaction streams into fixed-dimensional recurrent node states ($s_u(t)$). While effective for smooth temporal drift, this compression creates an uncharacterized vulnerability when structural interaction regimes return after extended conflicting dynamics ($\mathcal{A} \to \mathcal{B} \to \mathcal{A}$).

Standard temporal link prediction benchmarks evaluate models almost exclusively on continuous chronological sequences without regime reversals. This work addresses the foundational question:

> **"When a previously relevant temporal regime recurs, how much historical predictive information remains recoverable from a continuously updated recurrent representation?"**

---

## 📄 Manuscripts & Review Drafts

This repository maintains two versions of the manuscript:

| Version | Target Audience | Key Artifacts |
|:---|:---|:---|
| **[Final Publication Manuscript](temporal_recurrence/paper/final/)** | Conference Submission / Camera-Ready | • [`temporal_graph_recurrence_manuscript.pdf`](temporal_recurrence/paper/final/temporal_graph_recurrence_manuscript.pdf)<br>• [`main.tex`](temporal_recurrence/paper/final/main.tex)<br>• [`reproducibility_checklist.md`](temporal_recurrence/paper/final/reproducibility_checklist.md) |
| **[Professor-Review / Comparison Draft](temporal_recurrence/paper/professor_comparison/)** | Faculty Review & Pedagogical Comparison | • [`temporal_graph_recurrence_professor_comparison.pdf`](temporal_recurrence/paper/professor_comparison/temporal_graph_recurrence_professor_comparison.pdf) (12 pages, fixed float queuing, compact typography)<br>• [`main.tex`](temporal_recurrence/paper/professor_comparison/main.tex)<br>• [`comparison_notes.md`](temporal_recurrence/paper/professor_comparison/comparison_notes.md)<br>• [`appendix/supplementary_material.tex`](temporal_recurrence/paper/professor_comparison/appendix/supplementary_material.tex)<br>• [`references.bib`](temporal_recurrence/paper/professor_comparison/references.bib) (30 verified peer-reviewed citations) |

---

## 💡 Key Scientific Findings & Hypotheses

- **H1 (Duration-Dependent Interference):** Random historical retrieval exhibits monotonic degradation ($0.7423 \to 0.7193$ AP) as distractor duration $T_B$ increases from $25$ to $200$ snapshots on density-matched Dynamic SBMs ($\rho = 0.10$).
- **H2 (Capacity Limits):** Quadrupling recurrent memory dimensionality ($d_m = 64 \to 256$) increases trainable parameters by $+412\%$ ($45,697 \to 234,113$) while yielding $<0.008$ AP improvement at $T_B=100$, confirming state expansion alone cannot overcome sequential overwriting.
- **H3 (Addressable Historical Recovery):** Decoupling historical state preservation from continuous message passing via episodic key-value memory banks restores predictive utility at recurrence onset ($k_A = 0$).
- **H4 (Exact vs. Structural Recurrence):** Exact edge memorization (EdgeBank) excels on repeated pairwise edges ($\text{AP} = 0.8841$), but collapses under structural community recurrence ($\text{AP} = 0.5774$), where structural retrieval retains a statistically significant advantage ($+0.0351$ AP, $p < 10^{-6}$).
- **Cross-Domain Validation:** MA-TGN consistently outperforms continuous TGNs across 4 natural multi-week recurrence episodes in SNAP CollegeMsg ($\Delta\text{AP} = +0.0583$) and SNAP Bitcoin-OTC ($\Delta\text{AP} = +0.1243$).

---

## 📊 Full Research Paper Tables Registry

The complete collection of **14 empirical research tables** is published in the paper and accessible interactively in the webapp:

| Table # | Full Descriptive Scientific Title | Scope / Category | Primary Takeaway |
|:---:|:---|:---|:---|
| **Table 1** | **Systematic Literature Positioning and Methodological Comparison Across Dynamic Graph Paradigms** | *Taxonomy* | Situates work across Recurrent TGNNs, Transformers, Hash Tables, Continual Graph Learning, and Periodic Models. |
| **Table 2** | **Mathematical Notation and Canonical Benchmark Parameters** | *Protocol* | $N=300$, $K_{\text{comm}}=3$, $\rho=0.10$, $\lambda_A=0.35$, $\lambda_B=0.15$, $T_A=100$, $T_B\in\{25,50,100,200\}$. |
| **Table 3** | **Main Synthetic Benchmark Link Prediction Results Across Distractor Durations ($T_B$)** | *Synthetic Benchmark* | Mean $\pm$ Std across 10 random seeds ($42$–$51$) for Oracle, Heuristics, EdgeBank, and Neural models. |
| **Table 4** | **Continuous TGN Link Prediction AP across Memory Dimension ($d_m$) and Distractor Duration ($T_B$)** | *Capacity Scaling* | Evaluates $d_m \in [16, 256]$; proves capacity scaling does not resolve distractor overwriting. |
| **Table 5** | **Recurrence Decomposition: Exact Edge Repetition vs. Latent Structural Signal ($T_B=100$)** | *Decomposition* | Disentangles exact edge overlap from structural community recurrence ($\Delta\text{AP} = +0.0351, p < 10^{-6}$). |
| **Table 6** | **MA-TGN Architectural Component Ablation Matrix ($T_B=100$, 5 Seeds)** | *Ablation* | 7-model ablation (A–G); shows addressable episodic storage is the primary historical carrier. |
| **Table 7** | **Memory Addressing Mechanism and Key Routing Ablation ($T_B=100$, 5 Seeds)** | *Routing* | Evaluates multi-head attention, cosine key similarity, recency heuristics, and random retrieval. |
| **Table 8** | **Re-Exposure Dynamics and Adaptation Trajectory ($T_B=100$, 5 Seeds)** | *Adaptation* | Evaluates link prediction AP across renewed Regime $\mathcal{A}$ interactions ($k_A \in [0, 40]$). |
| **Table 9** | **Episode-by-Episode Real-World Recurrence Performance (SNAP CollegeMsg & SNAP Bitcoin-OTC)** | *Real-World* | MA-TGN exceeds Continuous TGN by $+0.0583$ AP (CollegeMsg) and $+0.1243$ AP (Bitcoin-OTC). |
| **Table 10** | **Analytical Memory Footprint and Model Parameter Accounting ($N=300, d_m=64$)** | *Complexity* | Single-precision float RAM ($827.5\text{ KB}$ for $K=10$), parameter counts, and inference latency ($184.2\,\mu\text{s}$). |
| **Table 11** | **Episodic Memory Checkpoint Capacity Sweep ($K \in \{1, \dots, 32\}$, $T_B=100$)** | *Scaling* | Memory footprint and performance saturation analysis across checkpoint bank sizes $K$. |
| **Table 12** | **Systematic Claim-to-Evidence Audit and Verification Matrix** | *Verification* | Audits all 6 scientific claims with empirical scopes and artifact mappings (all marked **PASS**). |
| **Table 13** | **Comprehensive Hyperparameter and Benchmark Experimental Configuration** | *Reproducibility* | Complete registry of hyperparameters, Adam optimizer parameters, and snapshot schedules. |
| **Table 14** | **Eight-Point Temporal Non-Anticipation and Causality Verification Protocol** | *Causality* | 8-point temporal non-anticipation leak verification protocol guaranteeing zero information leakage. |

---

## 🖥️ Interactive Simulation Dashboard & Frontend

A Vite + React 19 interactive developer interface is provided in [`temporal_recurrence/webapp`](temporal_recurrence/webapp):

- **Simulator & Topology View**: Real-time canvas graph visualization, interactive timeline scrubber, synchronized link-prediction AP curves, and episodic attention heatmaps.
- **Research Tables View (14 Tables)**: Categorized inspection of all 14 research tables with instant search filtering, monospace metrics, best-performer highlights, and one-click **Copy as Markdown** or **LaTeX** export.
- **Benchmark Audit View**: Phase-by-phase experimental verdict explorer and reproduction logs.

### Vercel Deployment Configuration
- **Framework Preset**: `Vite`
- **Root Directory**: `temporal_recurrence/webapp`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Install Command**: `npm install`
- **Environment Variable (Optional)**: `VITE_API_BASE` (e.g. `https://your-api-domain.com/api` if deploying FastAPI backend)

---

## 📁 Repository Architecture

```text
├── temporal_recurrence/
│   ├── paper/
│   │   ├── final/                  # Hardened camera-ready submission manuscript
│   │   │   ├── main.tex
│   │   │   ├── references.bib
│   │   │   ├── temporal_graph_recurrence_manuscript.pdf
│   │   │   └── reproducibility_checklist.md
│   │   └── professor_comparison/   # Pedagogical faculty-review comparison version
│   │       ├── main.tex
│   │       ├── references.bib
│   │       ├── comparison_notes.md
│   │       ├── appendix/supplementary_material.tex
│   │       └── temporal_graph_recurrence_professor_comparison.pdf
│   ├── src/
│   │   ├── baselines/              # EdgeBank, Current-Only, Historical Oracle
│   │   ├── evaluation/             # Non-anticipation leak audit & metric suite
│   │   ├── generator/              # Dynamic SBM benchmark generator (DSBM)
│   │   └── models/                 # Continuous TGN & Memory-Augmented TGN (MA-TGN)
│   ├── api/                        # FastAPI simulation backend for dashboard
│   ├── configs/                    # Canonical YAML benchmark configurations
│   ├── data/                       # SNAP CollegeMsg & Bitcoin-OTC episode splits
│   ├── experiments/                # Parameterized sweep execution scripts
│   ├── results/                    # Complete Phase 0 through Phase 12 CSV/JSON artifacts
│   ├── scripts/                    # Figure generators and evaluation pipelines
│   ├── tests/                      # Automated unit and integration test suite
│   └── webapp/                     # Interactive React/Vite dashboard & table registry
├── paper -> temporal_recurrence/paper
├── results -> temporal_recurrence/results
└── README.md
```

---

## 🚀 Installation & Quickstart

### 1. Clone Repository & Setup Python Environment
```bash
git clone https://github.com/PavanAksshay/temporal-memory-interference.git
cd temporal-memory-interference/temporal_recurrence

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
pytest tests/
```

### 3. Launch Frontend Web Dashboard
```bash
cd webapp
npm install
npm run dev
```

---

## 🛠️ Reproducing Paper Experiments & Compiling Manuscripts

### Run Empirical Benchmarks
```bash
# Execute Synthetic Recurrence Evaluation
bash scripts/run_phase4_5.sh

# Run Real-World Episode & Model Evaluation
python3 scripts/evaluate_real_and_matgn.py

# Regenerate Publication Figures
python3 scripts/generate_paper_figures.py
```

### Compile Manuscripts with Tectonic / pdflatex
```bash
# Build Professor-Review Comparison Draft
cd temporal_recurrence/paper/professor_comparison
tectonic main.tex --outdir .

# Build Final Submission Manuscript
cd ../final
tectonic main.tex --outdir .
```

---

## 📖 Citation

If you find this work or benchmark useful in your research, please cite:

```bibtex
@article{temporal_memory_interference2026,
  title={When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks},
  author={Anonymous Authors},
  journal={Empirical Temporal Graph Learning Working Group},
  year={2026}
}
```

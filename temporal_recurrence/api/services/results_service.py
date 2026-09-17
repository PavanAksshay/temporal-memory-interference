"""Service to discover, parse and serve experimental results, figures, and benchmark metrics."""
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def get_available_phases() -> List[Dict[str, Any]]:
    """Scans results directory and gathers available phases and their artifacts."""
    phases = []
    if not RESULTS_DIR.exists():
        return phases

    # Known phase metadata catalog
    phase_meta_map = {
        "phase0_1": {
            "title": "Phase 0/0.1: Synthetic Environment & Baseline Validation",
            "description": "Validates DSBM regime isolation, controlled density parity, and recency vs historical divergence."
        },
        "phase1": {
            "title": "Phase 1: TGNN Baseline Implementation & Memory Bias Audit",
            "description": "Continuous-time TGNN benchmarks (TGN, TGAT, JODIE) under recurrence $A \to B \to A$."
        },
        "phase1_1": {
            "title": "Phase 1.1: Recency Bias Formalization",
            "description": "Measures performance degradation as intervening duration $T_B$ increases."
        },
        "phase2": {
            "title": "Phase 2: Historical Memory Retrieval Architecture",
            "description": "Evaluates attention-based historical snapshot and episodic memory retrieval."
        },
        "phase3": {
            "title": "Phase 3: Cross-Domain & Generalization Benchmark",
            "description": "Evaluation on real-world datasets (CollegeMsg, Bitcoin-OTC, Wikipedia, Reddit)."
        },
        "phase4": {
            "title": "Phase 4: Publication Audit & Benchmark Suite",
            "description": "Rigorous ablation across 10 random seeds and statistical significance testing."
        },
        "phase4_5": {
            "title": "Phase 4.5: Final Scientific Falsification & Claim Verification",
            "description": "Independent stress test with hard benchmarks, random retrieval baselines, and capacity surface modeling."
        },
        "phase5": {
            "title": "Phase 5: Reviewer Risk Mitigation & Robustness",
            "description": "Addresses reviewer objections regarding leakage, density artifacts, and hyperparameter sensitivity."
        },
        "phase6": {
            "title": "Phase 6: Scaling & Long-Horizon Dynamics",
            "description": "Tests long-horizon memory limits ($T_B > 500$) and retrieval indexing bottlenecks."
        },
        "phase6_5": {
            "title": "Phase 6.5: Scientific Strengthening & Final Paper Artifacts",
            "description": "Consolidated LaTeX tables, camera-ready figures, and falsification reports."
        },
        "phase7": {
            "title": "Phase 7: Real-world Event Recurrence Scenarios",
            "description": "Seasonality and cyclic dynamics across multi-month temporal graphs."
        },
        "phase8": {
            "title": "Phase 8: Production Benchmark & Packaging",
            "description": "Final packaged datasets, checkpoints, and benchmark leaderboard."
        },
        "phase9": {
            "title": "Phase 9: Learnable Memory-Augmented TGNN (MA-TGN)",
            "description": "End-to-end differentiable episodic retrieval overcoming recency bias and recovery inertia."
        }
    }

    # Find phase directories
    for p_dir in sorted(RESULTS_DIR.iterdir()):
        if p_dir.is_dir() and (p_dir.name.startswith("phase") or p_dir.name == "figures"):
            p_name = p_dir.name
            figures = []
            fig_dir = p_dir / "figures" if p_dir.name != "figures" else p_dir
            if fig_dir.exists():
                for f in sorted(fig_dir.glob("*.png")):
                    figures.append({
                        "name": f.name,
                        "relative_path": f"{p_name}/figures/{f.name}" if p_name != "figures" else f"figures/{f.name}"
                    })

            reports = []
            rep_dir = p_dir / "reports"
            if rep_dir.exists():
                for r in sorted(rep_dir.glob("*.md")):
                    reports.append(r.name)

            has_verdict = False
            verdict_path = p_dir / "processed" / f"{p_name}_verdict.json"
            if verdict_path.exists():
                has_verdict = True

            has_csv = False
            proc_dir = p_dir / "processed"
            if proc_dir.exists() and list(proc_dir.glob("*.csv")):
                has_csv = True

            meta = phase_meta_map.get(p_name, {
                "title": f"Phase {p_name.replace('phase', '')}",
                "description": f"Experimental results and artifacts for {p_name}"
            })

            phases.append({
                "id": p_name,
                "title": meta["title"],
                "description": meta["description"],
                "num_figures": len(figures),
                "has_verdict": has_verdict,
                "has_csv": has_csv,
                "reports": reports
            })

    return phases


def get_phase_details(phase_id: str) -> Dict[str, Any]:
    """Retrieves full details, metrics, summary CSV data, and figures for a given phase."""
    p_dir = RESULTS_DIR / phase_id
    if not p_dir.exists():
        return {"error": f"Phase {phase_id} not found"}

    details: Dict[str, Any] = {
        "phase_id": phase_id,
        "figures": [],
        "verdict": None,
        "csv_tables": {},
        "reports": {}
    }

    # Figures
    fig_dir = p_dir / "figures" if phase_id != "figures" else p_dir
    if fig_dir.exists():
        for f in sorted(fig_dir.glob("*.png")):
            details["figures"].append({
                "name": f.name,
                "url": f"/api/figures/{phase_id}/{f.name}" if phase_id != "figures" else f"/api/figures/root/{f.name}"
            })

    # Verdict JSON
    proc_dir = p_dir / "processed"
    if proc_dir.exists():
        for vj in proc_dir.glob("*verdict.json"):
            try:
                with open(vj, "r") as f:
                    details["verdict"] = json.load(f)
            except Exception:
                pass

        # CSV tables (read first 20 rows of each for summary)
        for cf in proc_dir.glob("*.csv"):
            try:
                rows = []
                with open(cf, "r") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        if idx >= 25:
                            break
                        rows.append(row)
                details["csv_tables"][cf.name] = rows
            except Exception:
                pass

    # Reports
    rep_dir = p_dir / "reports"
    if rep_dir.exists():
        for rf in rep_dir.glob("*.md"):
            try:
                with open(rf, "r") as f:
                    details["reports"][rf.name] = f.read()[:3000] # preview
            except Exception:
                pass

    return details


def get_figure_path(phase_id: str, filename: str) -> Optional[Path]:
    """Finds absolute path of a figure PNG."""
    if phase_id == "root" or phase_id == "figures":
        p = RESULTS_DIR / "figures" / filename
    else:
        p = RESULTS_DIR / phase_id / "figures" / filename
    if p.exists() and p.is_file():
        return p
    return None

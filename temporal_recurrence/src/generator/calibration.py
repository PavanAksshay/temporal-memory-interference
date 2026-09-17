"""Calibration and parameter validation suite for regimes."""
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np

from .dsbm import DynamicSBMGenerator, DynamicGraphSequence
from .regimes import RegimeConfig, compute_regime_affinity_similarity


@dataclass
class RegimeStats:
    name: str
    target_density: float
    target_persistence: float
    empirical_mean_density: float
    empirical_density_std: float
    empirical_mean_degree: float
    empirical_degree_std: float
    empirical_autocorrelation: float
    within_comm_density: float
    between_comm_density: float


@dataclass
class CalibrationReport:
    regime_stats: Dict[str, RegimeStats]
    pairwise_similarities: Dict[str, float]  # e.g. "A_vs_B", "A_vs_C", "B_vs_C"
    density_diff_AB: float
    density_diff_AC: float
    max_density_diff_threshold: float
    passed: bool
    summary: str


class RegimeCalibrator:
    """Estimates empirical regime statistics, validates density parity, and measures structural overlap."""

    def __init__(
        self,
        generator: DynamicSBMGenerator,
        max_density_diff: float = 0.05,
        num_samples: int = 150,
        burn_in: int = 50
    ):
        self.generator = generator
        self.max_density_diff = max_density_diff
        self.num_samples = num_samples
        self.burn_in = burn_in

    def calibrate(self, seed: int = 42) -> CalibrationReport:
        stats: Dict[str, RegimeStats] = {}
        N = self.generator.num_nodes
        triu_idx = np.triu_indices(N, k=1)

        for regime_name, regime in self.generator.regimes.items():
            c = regime.community_assignments
            same_comm_mask = (c[triu_idx[0]] == c[triu_idx[1]])

            # Generate a single-regime sequence
            seq = self.generator.generate([(regime_name, self.burn_in + self.num_samples)], seed=seed)
            post_burn_snapshots = seq.snapshots[self.burn_in:]  # Shape: (S, N, N)

            # Measure densities and degrees over time
            densities = []
            within_densities = []
            between_densities = []
            degrees_all = []
            for G in post_burn_snapshots:
                upper = G[triu_idx]
                densities.append(float(np.mean(upper)))
                within_densities.append(float(np.mean(upper[same_comm_mask])) if np.any(same_comm_mask) else 0.0)
                between_densities.append(float(np.mean(upper[~same_comm_mask])) if np.any(~same_comm_mask) else 0.0)
                degrees_all.extend(np.sum(G, axis=1).tolist())

            # Measure temporal edge autocorrelation
            edge_trajectories = post_burn_snapshots[:, triu_idx[0], triu_idx[1]]  # Shape: (S, num_edges)
            X_t = edge_trajectories[:-1, :]
            X_tp1 = edge_trajectories[1:, :]

            mean_Xt = np.mean(X_t, axis=0)
            mean_Xtp1 = np.mean(X_tp1, axis=0)
            var_Xt = np.var(X_t, axis=0)
            var_Xtp1 = np.var(X_tp1, axis=0)

            valid_edges = (var_Xt > 1e-6) & (var_Xtp1 > 1e-6)
            if np.any(valid_edges):
                cov = np.mean((X_t[:, valid_edges] - mean_Xt[valid_edges]) * (X_tp1[:, valid_edges] - mean_Xtp1[valid_edges]), axis=0)
                corr = cov / np.sqrt(var_Xt[valid_edges] * var_Xtp1[valid_edges])
                emp_autocorr = float(np.mean(corr))
            else:
                emp_autocorr = 0.0

            stats[regime_name] = RegimeStats(
                name=regime_name,
                target_density=regime.target_density,
                target_persistence=regime.persistence,
                empirical_mean_density=float(np.mean(densities)),
                empirical_density_std=float(np.std(densities)),
                empirical_mean_degree=float(np.mean(degrees_all)),
                empirical_degree_std=float(np.std(degrees_all)),
                empirical_autocorrelation=emp_autocorr,
                within_comm_density=float(np.mean(within_densities)),
                between_comm_density=float(np.mean(between_densities))
            )

        # Pairwise structural similarities
        pairwise_sims = {}
        regimes = self.generator.regimes
        if "A" in regimes and "B" in regimes:
            pairwise_sims["A_vs_B"] = compute_regime_affinity_similarity(regimes["A"], regimes["B"])
        if "A" in regimes and "C" in regimes:
            pairwise_sims["A_vs_C"] = compute_regime_affinity_similarity(regimes["A"], regimes["C"])
        if "B" in regimes and "C" in regimes:
            pairwise_sims["B_vs_C"] = compute_regime_affinity_similarity(regimes["B"], regimes["C"])

        density_A = stats["A"].empirical_mean_density if "A" in stats else 0.0
        density_B = stats["B"].empirical_mean_density if "B" in stats else 0.0
        density_C = stats["C"].empirical_mean_density if "C" in stats else 0.0
        density_diff_AB = abs(density_A - density_B)
        density_diff_AC = abs(density_A - density_C)
        passed = (density_diff_AB <= self.max_density_diff) and (density_diff_AC <= self.max_density_diff)

        summary_lines = [
            f"Calibration Report (Seed: {seed}):",
            f"  Regime A: density={density_A:.4f}, mean_degree={stats['A'].empirical_mean_degree:.2f}, autocorr={stats['A'].empirical_autocorrelation:.4f}",
            f"  Regime B: density={density_B:.4f}, mean_degree={stats['B'].empirical_mean_degree:.2f}, autocorr={stats['B'].empirical_autocorrelation:.4f}",
            f"  Regime C: density={density_C:.4f}, mean_degree={stats['C'].empirical_mean_degree:.2f}, autocorr={stats['C'].empirical_autocorrelation:.4f}",
            f"  Pairwise Cosine Similarities: A-vs-B={pairwise_sims.get('A_vs_B', 0.0):.4f}, A-vs-C={pairwise_sims.get('A_vs_C', 0.0):.4f}, B-vs-C={pairwise_sims.get('B_vs_C', 0.0):.4f}",
            f"  Density differences: |A-B|={density_diff_AB:.4f}, |A-C|={density_diff_AC:.4f} (threshold <= {self.max_density_diff:.4f})",
            f"  Calibration status: {'PASSED' if passed else 'FAILED'}"
        ]

        return CalibrationReport(
            regime_stats=stats,
            pairwise_similarities=pairwise_sims,
            density_diff_AB=density_diff_AB,
            density_diff_AC=density_diff_AC,
            max_density_diff_threshold=self.max_density_diff,
            passed=passed,
            summary="\n".join(summary_lines)
        )

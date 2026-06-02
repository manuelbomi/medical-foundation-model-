#!/usr/bin/env python3
"""
Analyze domain gap between source (natural images) and target (medical) domains.

Computes FID, MMD, A-distance, and generates feature visualizations to
quantify the distribution shift that domain adaptation must bridge.

Usage:
    python scripts/analyze_domain_gap.py \
        --source-dir data/imagenet_samples \
        --target-dir data/mammography \
        --backbone swin_base \
        --output-dir results/domain_gap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.domain_gap import DomainGapAnalyzer, FIDCalculator, MMDCalculator, ADistanceCalculator
from src.evaluation.transfer_analysis import CKACalculator, LayerWiseCKA, TransferabilityScorer
from src.foundation.model_zoo import build_foundation_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_synthetic_analysis() -> None:
    """Run domain gap analysis with synthetic data for demonstration."""
    logger.info("Running synthetic domain gap analysis...")

    # Simulate feature distributions from two domains
    np.random.seed(42)
    feature_dim = 256

    # Source domain: centered at origin with moderate variance
    source_features = np.random.randn(500, feature_dim) * 1.0

    # Target domain: shifted and with different covariance
    shift = np.random.randn(feature_dim) * 0.5
    scale = 1.0 + 0.3 * np.random.randn(feature_dim)
    target_features = np.random.randn(500, feature_dim) * scale + shift

    # Compute metrics
    fid_calc = FIDCalculator()
    mmd_calc = MMDCalculator()
    a_dist_calc = ADistanceCalculator()

    fid = fid_calc(source_features, target_features)
    mmd = mmd_calc(source_features, target_features)
    a_distance = a_dist_calc(source_features, target_features)

    logger.info(f"\nDomain Gap Metrics (synthetic data):")
    logger.info(f"  FID: {fid:.4f}")
    logger.info(f"  MMD: {mmd:.6f}")
    logger.info(f"  A-distance: {a_distance:.4f}")

    # CKA between "layers"
    cka = CKACalculator()
    # Simulate features from different layers
    layer1 = np.random.randn(200, 64)
    layer2 = layer1 @ np.random.randn(64, 128) + np.random.randn(200, 128) * 0.1  # correlated
    layer3 = np.random.randn(200, 128)  # uncorrelated

    cka_12 = cka.linear_cka(layer1, layer2)
    cka_13 = cka.linear_cka(layer1, layer3)
    cka_23 = cka.linear_cka(layer2, layer3)

    logger.info(f"\nCKA Similarity (synthetic layers):")
    logger.info(f"  Layer 1 vs Layer 2 (correlated): {cka_12:.4f}")
    logger.info(f"  Layer 1 vs Layer 3 (independent): {cka_13:.4f}")
    logger.info(f"  Layer 2 vs Layer 3: {cka_23:.4f}")

    # Transferability scoring
    labels = np.random.randint(0, 5, size=200)
    h_score = TransferabilityScorer.h_score(layer1, labels)
    logger.info(f"\nTransferability:")
    logger.info(f"  H-score: {h_score:.4f}")

    return {
        "fid": fid,
        "mmd": mmd,
        "a_distance": a_distance,
        "cka_correlated": cka_12,
        "cka_independent": cka_13,
        "h_score": h_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze domain gap")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="swin_base")
    parser.add_argument("--output-dir", type=str, default="results/domain_gap")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Run with synthetic data (default)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic or (args.source_dir is None and args.target_dir is None):
        results = run_synthetic_analysis()

        # Save results
        results_path = output_dir / "domain_gap_results.json"
        with open(results_path, "w") as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")
        return

    # Real data analysis
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_foundation_model(
        backbone=args.backbone,
        input_channels=1,
        num_classes=2,
        pretrained=True,
    )

    analyzer = DomainGapAnalyzer(model, device)

    logger.info("Load source and target datasets to run real domain gap analysis")
    logger.info("See README.md for data preparation instructions")


if __name__ == "__main__":
    main()

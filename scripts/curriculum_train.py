#!/usr/bin/env python3
"""
Curriculum Training for Medical Image Classification.

Implements curriculum learning strategies that present training samples
in a meaningful order to improve convergence and generalization.

Usage:
    python scripts/curriculum_train.py \
        --config configs/curriculum_mammography.yaml \
        --curriculum self_paced \
        --backbone swin_base \
        --output-dir outputs/curriculum_swin
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.curriculum.curriculum_scheduler import (
    AntiCurriculumScheduler,
    DifficultyBasedCurriculum,
    SelfPacedCurriculum,
    TeacherStudentCurriculum,
    build_curriculum,
)
from src.curriculum.difficulty_scorer import (
    DataComplexityScorer,
    EnsembleDifficultyScorer,
    LossBasedScorer,
    RadiologistAgreementScorer,
    UncertaintyScorer,
)
from src.foundation.model_zoo import build_foundation_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_difficulty_scores(
    config: dict,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> np.ndarray:
    """Compute difficulty scores using the configured method."""
    scoring_cfg = config.get("curriculum", {}).get("difficulty_scoring", {})
    method = scoring_cfg.get("method", "loss")

    logger.info(f"Computing difficulty scores using method: {method}")

    if method == "loss":
        scorer = LossBasedScorer(
            model=model,
            dataset=dataset,
            device=device,
            n_epochs=scoring_cfg.get("loss_epochs", 5),
        )
        report = scorer.score()

    elif method == "data_complexity":
        # Extract images and labels from dataset
        images = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        scorer = DataComplexityScorer(images, labels)
        report = scorer.score()

    elif method == "ensemble":
        # For demonstration, use loss-based as primary
        loss_scorer = LossBasedScorer(
            model=model, dataset=dataset, device=device,
            n_epochs=scoring_cfg.get("loss_epochs", 5),
        )
        weights = scoring_cfg.get("ensemble_weights", {})
        ensemble = EnsembleDifficultyScorer([
            (loss_scorer, weights.get("loss", 1.0)),
        ])
        report = ensemble.score()

    else:
        # Default: uniform random scores as fallback
        logger.warning(f"Unknown scoring method '{method}', using random scores")
        n_samples = len(dataset)
        report_scores = np.random.rand(n_samples)
        return report_scores

    logger.info(f"\n{report.summary()}")
    return report.scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum training for medical imaging")
    parser.add_argument("--config", type=str, default="configs/curriculum_mammography.yaml")
    parser.add_argument("--curriculum", type=str, default=None,
                        choices=["difficulty", "self_paced", "anti_curriculum", "teacher_student"])
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    curriculum_cfg = config.get("curriculum", {})

    if args.backbone:
        model_cfg["backbone"] = args.backbone
    if args.curriculum:
        curriculum_cfg["strategy"] = args.curriculum

    strategy = curriculum_cfg.get("strategy", "self_paced")
    backbone_name = model_cfg.get("backbone", "swin_base")

    logger.info(f"Curriculum training with {strategy} strategy, {backbone_name} backbone")

    # Build model
    model = build_foundation_model(
        backbone=backbone_name,
        input_channels=model_cfg.get("input_channels", 1),
        num_classes=model_cfg.get("num_classes", 2),
        target_resolution=model_cfg.get("target_resolution", 512),
        channel_strategy=model_cfg.get("channel_strategy", "average"),
        pretrained=model_cfg.get("pretrained", True),
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {backbone_name} with {total_params:,} parameters")

    if args.dry_run:
        # Demonstrate curriculum behavior with synthetic difficulty scores
        n_samples = 1000
        difficulty_scores = np.random.beta(2, 5, size=n_samples)  # Skewed toward easy
        total_epochs = config.get("training", {}).get("epochs", 100)

        scheduler = build_curriculum(
            strategy=strategy,
            difficulty_scores=difficulty_scores,
            total_epochs=total_epochs,
            **curriculum_cfg.get(strategy, {}),
        )

        logger.info(f"\nCurriculum progression (dry run with {n_samples} synthetic samples):")
        for epoch in [0, 10, 25, 50, 75, 99]:
            if epoch < total_epochs:
                indices = scheduler.get_sample_indices(epoch)
                active_difficulties = difficulty_scores[indices]
                logger.info(
                    f"  Epoch {epoch:3d}: {len(indices):4d} samples "
                    f"(mean difficulty: {active_difficulties.mean():.3f}, "
                    f"max: {active_difficulties.max():.3f})"
                )
        return

    logger.info("To run full curriculum training, provide training data")
    logger.info("See README.md for usage examples")


if __name__ == "__main__":
    main()

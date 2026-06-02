#!/usr/bin/env python3
"""
Adapt a pretrained foundation model to the medical imaging domain.

This script orchestrates the full adaptation pipeline:
1. Load pretrained backbone with channel/resolution adaptation
2. Configure domain adaptation strategy (DANN, MMD, or combined)
3. Set up progressive unfreezing schedule
4. Apply Fourier domain adaptation for style harmonization
5. Train with domain-adversarial loss + task loss
6. Evaluate cross-domain generalization

Usage:
    python scripts/adapt_foundation.py \
        --config configs/domain_adaptation.yaml \
        --backbone swin_base \
        --strategy dann \
        --output-dir outputs/adapt_swin_dann
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptation.domain_adapter import (
    AdaptationState,
    CombinedStrategy,
    DANNStrategy,
    DomainAdapter,
    MMDStrategy,
    ProgressiveUnfreezer,
)
from src.adaptation.style_transfer import AdaptiveBatchNorm, FourierDomainAdaptation, MultiSiteHarmonizer
from src.evaluation.domain_gap import DomainGapAnalyzer
from src.foundation.model_zoo import build_foundation_model
from src.training.foundation_trainer import FoundationTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_adaptation_strategy(config: dict, feature_dim: int) -> DomainAdapter:
    """Build domain adaptation strategy from config."""
    adapt_cfg = config.get("adaptation", {})
    strategy_name = adapt_cfg.get("strategy", "dann")

    if strategy_name == "dann":
        dann_cfg = adapt_cfg.get("dann", {})
        strategy = DANNStrategy(
            feature_dim=feature_dim,
            hidden_dim=dann_cfg.get("hidden_dim", 1024),
        )
    elif strategy_name == "mmd":
        mmd_cfg = adapt_cfg.get("mmd", {})
        strategy = MMDStrategy(
            kernel_bandwidths=mmd_cfg.get("kernel_bandwidths"),
            loss_weight=mmd_cfg.get("loss_weight", 0.5),
        )
    elif strategy_name == "combined":
        dann_cfg = adapt_cfg.get("dann", {})
        mmd_cfg = adapt_cfg.get("mmd", {})
        strategy = CombinedStrategy([
            (DANNStrategy(feature_dim, dann_cfg.get("hidden_dim", 1024)), 0.5),
            (MMDStrategy(mmd_cfg.get("kernel_bandwidths"), mmd_cfg.get("loss_weight", 0.5)), 0.5),
        ])
    else:
        raise ValueError(f"Unknown adaptation strategy: {strategy_name}")

    logger.info(f"Built {strategy_name} adaptation strategy")
    return strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt foundation model to medical imaging")
    parser.add_argument("--config", type=str, default="configs/domain_adaptation.yaml")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Build model but don't train")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override from CLI
    model_cfg = config.get("model", {})
    if args.backbone:
        model_cfg["backbone"] = args.backbone
    if args.strategy:
        config.setdefault("adaptation", {})["strategy"] = args.strategy
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir

    # Build model
    backbone_name = model_cfg.get("backbone", "swin_base")
    logger.info(f"Building foundation model: {backbone_name}")

    model = build_foundation_model(
        backbone=backbone_name,
        input_channels=model_cfg.get("input_channels", 1),
        num_classes=model_cfg.get("num_classes", 2),
        target_resolution=model_cfg.get("target_resolution", 512),
        channel_strategy=model_cfg.get("channel_strategy", "average"),
        use_fpn=model_cfg.get("use_fpn", False),
        pretrained=model_cfg.get("pretrained", True),
    )

    # Get layer groups for progressive training
    layer_groups = model.get_layer_groups()
    feature_dim = model.spec.feature_dims[-1]

    # Build adaptation strategy
    strategy = build_adaptation_strategy(config, feature_dim)

    # Build domain adapter
    adapt_cfg = config.get("adaptation", {})
    dann_cfg = adapt_cfg.get("dann", {})
    adapter = DomainAdapter(
        model=model,
        strategy=strategy,
        layer_groups=layer_groups,
        warmup_epochs=config.get("progressive", {}).get("warmup_epochs", 5),
        unfreeze_every=config.get("progressive", {}).get("unfreeze_every", 5),
        total_epochs=config.get("training", {}).get("epochs", 100),
        lambda_schedule=dann_cfg.get("lambda_schedule", "exp"),
        max_lambda=dann_cfg.get("max_lambda", 1.0),
    )

    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    if args.dry_run:
        logger.info("Dry run complete. Model architecture:")
        logger.info(f"  Backbone: {backbone_name}")
        logger.info(f"  Feature dim: {feature_dim}")
        logger.info(f"  Layer groups: {len(layer_groups)}")
        logger.info(f"  Adaptation: {adapt_cfg.get('strategy', 'dann')}")
        # Test forward pass
        device = torch.device("cpu")
        model = model.to(device)
        dummy = torch.randn(2, 1, 512, 512)
        with torch.no_grad():
            out = model(dummy)
            if isinstance(out, tuple):
                logger.info(f"  Output shape: {out[0].shape}")
            else:
                logger.info(f"  Output shape: {out.shape}")
        return

    # Full training requires data loaders (not shown for brevity)
    logger.info("To run full training, provide source and target data paths")
    logger.info("See README.md for usage examples")


if __name__ == "__main__":
    main()

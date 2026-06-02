"""
Unified Training Framework for Medical Image Foundation Models.

Orchestrates domain adaptation, curriculum learning, progressive transfer,
and few-shot learning into a single coherent training pipeline with
comprehensive logging and checkpointing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.adaptation.domain_adapter import AdaptationState, DomainAdapter, DomainAdaptationStrategy
from src.curriculum.curriculum_scheduler import CurriculumScheduler
from src.curriculum.difficulty_scorer import DifficultyScorer, LossBasedScorer
from src.transfer.progressive_training import (
    GradualUnfreezingScheduler,
    LayerWiseLRScheduler,
    ProgressiveResolutionTrainer,
    ProgressiveTrainingPipeline,
    ResolutionSchedule,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the unified training framework."""
    # General
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "outputs"

    # Optimization
    base_lr: float = 1e-3
    weight_decay: float = 0.01
    lr_decay_factor: float = 0.65
    warmup_epochs: int = 5
    min_lr: float = 1e-7

    # Mixed precision
    use_amp: bool = True

    # Domain adaptation
    use_adaptation: bool = True
    adaptation_strategy: str = "dann"  # dann, mmd, combined
    adaptation_weight: float = 1.0
    lambda_schedule: str = "exp"
    max_lambda: float = 1.0

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_strategy: str = "self_paced"
    initial_fraction: float = 0.3

    # Progressive training
    use_progressive: bool = True
    unfreeze_every: int = 5
    use_resolution_scaling: bool = True

    # Knowledge distillation
    use_distillation: bool = False
    distill_temperature: float = 4.0
    distill_alpha: float = 0.7

    # Logging
    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 10
    patience: int = 20


@dataclass
class TrainingMetrics:
    """Accumulated metrics for a training epoch."""
    loss: float = 0.0
    task_loss: float = 0.0
    adaptation_loss: float = 0.0
    distillation_loss: float = 0.0
    accuracy: float = 0.0
    n_samples: int = 0
    epoch_time: float = 0.0

    def update(self, losses: Dict[str, torch.Tensor], preds: torch.Tensor, labels: torch.Tensor) -> None:
        bs = labels.size(0)
        self.n_samples += bs
        self.loss += losses.get("total_loss", torch.tensor(0.0)).item() * bs
        self.task_loss += losses.get("task_loss", torch.tensor(0.0)).item() * bs
        self.adaptation_loss += losses.get("adaptation_loss", torch.tensor(0.0)).item() * bs
        self.distillation_loss += losses.get("kd_loss", torch.tensor(0.0)).item() * bs
        self.accuracy += (preds.argmax(dim=-1) == labels).float().sum().item()

    def compute(self) -> Dict[str, float]:
        n = max(self.n_samples, 1)
        return {
            "loss": self.loss / n,
            "task_loss": self.task_loss / n,
            "adaptation_loss": self.adaptation_loss / n,
            "distillation_loss": self.distillation_loss / n,
            "accuracy": self.accuracy / n,
            "epoch_time": self.epoch_time,
        }


class FoundationTrainer:
    """Unified trainer combining domain adaptation, curriculum learning,
    progressive transfer, and knowledge distillation.

    This is the main entry point for training foundation models on medical
    imaging tasks. It orchestrates all strategies and handles the training
    loop, evaluation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        domain_adapter: Optional[DomainAdapter] = None,
        curriculum: Optional[CurriculumScheduler] = None,
        progressive_pipeline: Optional[ProgressiveTrainingPipeline] = None,
        source_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.domain_adapter = domain_adapter
        self.curriculum = curriculum
        self.progressive = progressive_pipeline
        self.source_loader = source_loader

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup optimizer
        if self.progressive is not None:
            self.optimizer = self.progressive.setup_optimizer()
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay
            )

        # Cosine annealing scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.min_lr
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Tracking
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history: List[Dict[str, float]] = []

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FoundationTrainer initialized on {self.device}")
        logger.info(f"  Adaptation: {config.use_adaptation} ({config.adaptation_strategy})")
        logger.info(f"  Curriculum: {config.use_curriculum} ({config.curriculum_strategy})")
        logger.info(f"  Progressive: {config.use_progressive}")
        logger.info(f"  Distillation: {config.use_distillation}")

    def train(self) -> Dict[str, Any]:
        """Run the full training loop."""
        logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            # Update strategies
            self._on_epoch_start(epoch)

            # Train
            train_metrics = self._train_epoch(epoch)

            # Evaluate
            val_metrics = self._validate(epoch)

            # Log
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Learning rate step
            self.lr_scheduler.step()

            # Checkpointing and early stopping
            improved = self._check_improvement(val_metrics)
            if not improved:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                self.patience_counter = 0
                if epoch % self.config.save_interval == 0:
                    self._save_checkpoint(epoch, val_metrics)

        # Save final model
        self._save_checkpoint(epoch, val_metrics, final=True)

        return {
            "best_val_metric": self.best_val_metric,
            "final_epoch": epoch,
            "history": self.history,
        }

    def _on_epoch_start(self, epoch: int) -> None:
        """Update all strategies at epoch boundary."""
        if self.domain_adapter is not None:
            self.domain_adapter.on_epoch_start(epoch)

        if self.curriculum is not None:
            self.curriculum.step(epoch)

        if self.progressive is not None:
            config = self.progressive.on_epoch_start(epoch)
            logger.debug(f"Progressive config for epoch {epoch}: {config}")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        metrics = TrainingMetrics()
        start_time = time.time()

        # Use curriculum sampler if available
        if self.curriculum is not None:
            loader = DataLoader(
                self.train_loader.dataset,
                batch_size=self.config.batch_size,
                sampler=self.curriculum.get_sampler(epoch),
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loader = self.train_loader

        source_iter = iter(self.source_loader) if self.source_loader else None

        for batch_idx, batch in enumerate(loader):
            losses = self._train_step(batch, source_iter, epoch)

            # Extract predictions for metrics
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            with torch.no_grad():
                preds = self.model(images)
                if isinstance(preds, tuple):
                    preds = preds[0]
            metrics.update(losses, preds, labels)

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg = metrics.compute()
                logger.debug(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                    f"loss={avg['loss']:.4f} acc={avg['accuracy']:.4f}"
                )

        metrics.epoch_time = time.time() - start_time
        return metrics.compute()

    def _train_step(
        self,
        batch: Any,
        source_iter: Any,
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """Single training step with all strategies applied."""
        images = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        # Resolution scaling
        if self.progressive and self.progressive.resolution_trainer:
            resolution = self.progressive.resolution_trainer.current_resolution
            images = self.progressive.resolution_trainer.resize_batch(images, resolution)

        self.optimizer.zero_grad()

        use_amp = self.config.use_amp and self.scaler is not None

        with autocast(enabled=use_amp):
            # Forward pass
            output = self.model(images, return_features=True)
            if isinstance(output, tuple):
                logits, features = output
            else:
                logits = output
                features = None

            # Task loss (possibly with distillation)
            if self.progressive and self.progressive.distillation is not None:
                losses = self.progressive.compute_loss(
                    logits, labels, images=images, student_features=features
                )
            else:
                task_loss = F.cross_entropy(logits, labels)
                losses = {"total_loss": task_loss, "task_loss": task_loss}

            # Domain adaptation loss
            if self.domain_adapter is not None and source_iter is not None:
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.source_loader)
                    source_batch = next(source_iter)

                source_images = source_batch[0].to(self.device)
                with torch.no_grad():
                    source_out = self.model(source_images, return_features=True)
                    if isinstance(source_out, tuple):
                        _, source_features = source_out
                    else:
                        source_features = [source_out]

                if features is not None and source_features:
                    adapt_loss = self.domain_adapter.compute_loss(
                        source_features[-1], features[-1]
                    )
                    losses["adaptation_loss"] = adapt_loss
                    losses["total_loss"] = (
                        losses["total_loss"] + self.config.adaptation_weight * adapt_loss
                    )

        # Backward pass
        total_loss = losses["total_loss"]
        if use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return {k: v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in losses.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        for batch in self.val_loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            logits = self.model(images)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        val_loss = total_loss / max(total, 1)
        val_acc = correct / max(total, 1)

        # Compute AUC if binary classification
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        val_auc = self._compute_auc(all_probs, all_labels)

        return {
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
        }

    def _compute_auc(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute AUC for binary or multiclass classification."""
        if probs.shape[1] == 2:
            # Binary: use positive class probability
            pos_probs = probs[:, 1].numpy()
            true_labels = labels.numpy()

            # Simple AUC via Wilcoxon-Mann-Whitney
            pos_scores = pos_probs[true_labels == 1]
            neg_scores = pos_probs[true_labels == 0]

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                return 0.5

            auc = 0.0
            for p in pos_scores:
                auc += (neg_scores < p).sum() + 0.5 * (neg_scores == p).sum()
            auc /= len(pos_scores) * len(neg_scores)
            return float(auc)

        # Multiclass: one-vs-rest average (simplified)
        return float(probs.argmax(dim=-1).eq(labels).float().mean())

    def _check_improvement(self, val_metrics: Dict[str, float]) -> bool:
        """Check if validation metrics improved."""
        metric = val_metrics.get("val_auc", val_metrics.get("val_accuracy", 0.0))
        if metric > self.best_val_metric:
            self.best_val_metric = metric
            return True
        return False

    def _log_epoch(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ) -> None:
        """Log epoch results."""
        combined = {**train_metrics, **val_metrics, "epoch": epoch}
        self.history.append(combined)

        current_lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_acc={val_metrics['val_accuracy']:.4f} "
            f"val_auc={val_metrics.get('val_auc', 0):.4f} | "
            f"lr={current_lr:.2e} "
            f"time={train_metrics['epoch_time']:.1f}s"
        )

    def _save_checkpoint(
        self, epoch: int, val_metrics: Dict[str, float], final: bool = False
    ) -> None:
        """Save model checkpoint."""
        tag = "final" if final else f"epoch_{epoch}"
        path = self.output_dir / f"checkpoint_{tag}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_val_metric": self.best_val_metric,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

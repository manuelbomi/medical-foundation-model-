"""
Progressive Training Strategies for Medical Image Transfer Learning.

Implements techniques for gradually transitioning a pretrained model to
the target medical imaging domain, including resolution scaling, layer-wise
learning rate decay, gradual unfreezing, and knowledge distillation.

These strategies are crucial for medical imaging because:
1. Medical images have much higher resolution than ImageNet (e.g., 2048x2048 mammograms)
2. Low-level features (edges, textures) transfer well; high-level features need adaptation
3. Small dataset sizes make catastrophic forgetting a real risk
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution Scaling
# ---------------------------------------------------------------------------

@dataclass
class ResolutionSchedule:
    """Schedule for progressive resolution scaling during training.

    Start training at low resolution (fast, coarse features) and gradually
    increase to full resolution (slow, fine details). This provides:
    - Faster early epochs (more iterations per GPU-hour)
    - Natural curriculum: low-res images are "easier" (less fine detail)
    - Better optimization landscape early on
    """
    stages: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 224),    # epochs 0-N: 224x224
        (20, 384),   # epochs 20-N: 384x384
        (40, 512),   # epochs 40-N: 512x512
        (60, 768),   # epochs 60+: 768x768
    ])

    def get_resolution(self, epoch: int) -> int:
        """Get the target resolution for the current epoch."""
        resolution = self.stages[0][1]
        for start_epoch, res in self.stages:
            if epoch >= start_epoch:
                resolution = res
        return resolution


class ProgressiveResolutionTrainer:
    """Manages resolution scaling during training.

    Handles resizing transforms and adjusts batch size inversely with
    resolution to maintain constant GPU memory usage.
    """

    def __init__(
        self,
        schedule: ResolutionSchedule,
        base_batch_size: int = 32,
        base_resolution: int = 224,
        max_resolution: int = 1024,
    ) -> None:
        self.schedule = schedule
        self.base_batch_size = base_batch_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.current_resolution = base_resolution

    def get_resolution_and_batch_size(self, epoch: int) -> Tuple[int, int]:
        """Get resolution and adjusted batch size for the epoch."""
        resolution = self.schedule.get_resolution(epoch)
        resolution = min(resolution, self.max_resolution)

        # Scale batch size inversely with resolution^2 (to maintain memory)
        scale_factor = (self.base_resolution / resolution) ** 2
        batch_size = max(1, int(self.base_batch_size * scale_factor))

        if resolution != self.current_resolution:
            logger.info(
                f"Resolution change at epoch {epoch}: {self.current_resolution} -> {resolution}, "
                f"batch_size: {batch_size}"
            )
            self.current_resolution = resolution

        return resolution, batch_size

    @staticmethod
    def resize_batch(
        images: torch.Tensor, target_size: int, mode: str = "bilinear"
    ) -> torch.Tensor:
        """Resize a batch of images to the target resolution."""
        if images.shape[-1] == target_size and images.shape[-2] == target_size:
            return images
        return F.interpolate(
            images,
            size=(target_size, target_size),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )


# ---------------------------------------------------------------------------
# Layer-wise Learning Rate Decay
# ---------------------------------------------------------------------------

class LayerWiseLRScheduler:
    """Assign different learning rates to different layer groups.

    Lower layers get smaller LR (they contain transferable features),
    higher layers get larger LR (they need more adaptation). This is
    parameterized as exponential decay from the classifier head backward.

    For a model with N layer groups:
      LR(group_i) = base_lr * decay_factor^(N - 1 - i)

    The classifier head (group N-1) gets base_lr, and the first conv
    (group 0) gets base_lr * decay_factor^(N-1).
    """

    def __init__(
        self,
        layer_groups: List[List[nn.Parameter]],
        base_lr: float = 1e-3,
        decay_factor: float = 0.65,
        min_lr: float = 1e-7,
    ) -> None:
        self.layer_groups = layer_groups
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.n_groups = len(layer_groups)

    def get_param_groups(self) -> List[Dict[str, Any]]:
        """Build optimizer parameter groups with layer-wise learning rates."""
        param_groups = []
        for i, group_params in enumerate(self.layer_groups):
            if not group_params:
                continue
            # Exponential decay: highest LR for last group, lowest for first
            lr = self.base_lr * (self.decay_factor ** (self.n_groups - 1 - i))
            lr = max(lr, self.min_lr)
            param_groups.append({
                "params": group_params,
                "lr": lr,
                "name": f"layer_group_{i}",
            })

        # Log the LR schedule
        for pg in param_groups:
            n_params = sum(p.numel() for p in pg["params"])
            logger.info(f"  {pg['name']}: lr={pg['lr']:.2e}, params={n_params:,}")

        return param_groups


# ---------------------------------------------------------------------------
# Gradual Unfreezing Scheduler
# ---------------------------------------------------------------------------

class GradualUnfreezingScheduler:
    """Gradually unfreeze model layers during fine-tuning.

    Starts with all backbone layers frozen (linear probing phase) and
    unfreezes one layer group at a time from top (near classifier) to
    bottom (near input). Each newly unfrozen group enters with a warmup
    period at reduced LR.

    This is critical for medical imaging where:
    - Low-level features (edges, textures) transfer well and should be preserved
    - The dataset is small, so unfreezing everything at once risks overfitting
    - Gradual unfreezing acts as implicit regularization
    """

    def __init__(
        self,
        model: nn.Module,
        layer_groups: List[List[nn.Parameter]],
        optimizer: Optimizer,
        warmup_epochs: int = 3,
        unfreeze_every: int = 5,
        lr_warmup_factor: float = 0.1,
    ) -> None:
        self.model = model
        self.layer_groups = layer_groups
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.unfreeze_every = unfreeze_every
        self.lr_warmup_factor = lr_warmup_factor
        self.n_groups = len(layer_groups)

        # Start with everything frozen except the last group (classifier)
        self.unfrozen_from = self.n_groups - 1
        self._apply_freeze_state()

        # Track warmup epochs per group
        self.group_unfreeze_epoch: Dict[int, int] = {self.n_groups - 1: 0}

    def _apply_freeze_state(self) -> None:
        for i, group in enumerate(self.layer_groups):
            requires_grad = i >= self.unfrozen_from
            for param in group:
                param.requires_grad = requires_grad

    def step(self, epoch: int) -> bool:
        """Check if a new group should be unfrozen. Returns True if state changed."""
        if self.unfrozen_from <= 0:
            return False  # Everything already unfrozen

        if epoch < self.warmup_epochs:
            return False  # Still in initial warmup

        adjusted = epoch - self.warmup_epochs
        target_unfrozen = self.n_groups - 1 - (adjusted // self.unfreeze_every)
        target_unfrozen = max(0, target_unfrozen)

        if target_unfrozen < self.unfrozen_from:
            self.unfrozen_from = target_unfrozen
            self._apply_freeze_state()
            self.group_unfreeze_epoch[self.unfrozen_from] = epoch

            logger.info(
                f"Epoch {epoch}: unfroze layer group {self.unfrozen_from}, "
                f"trainable params: {self._count_trainable():,}"
            )
            return True
        return False

    def get_lr_multiplier(self, group_idx: int, epoch: int) -> float:
        """Get LR multiplier accounting for per-group warmup."""
        if group_idx not in self.group_unfreeze_epoch:
            return 0.0

        epochs_since_unfreeze = epoch - self.group_unfreeze_epoch[group_idx]
        if epochs_since_unfreeze < self.warmup_epochs:
            # Linear warmup from lr_warmup_factor to 1.0
            progress = epochs_since_unfreeze / self.warmup_epochs
            return self.lr_warmup_factor + (1.0 - self.lr_warmup_factor) * progress
        return 1.0

    def _count_trainable(self) -> int:
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------

class KnowledgeDistillation(nn.Module):
    """Knowledge distillation from a larger teacher model to a smaller student.

    Transfers dark knowledge (soft label distributions) from a high-capacity
    teacher to a compact student model. In the medical imaging context, the
    teacher might be:
    - A larger architecture (e.g., ViT-L -> ViT-B)
    - An ensemble of models
    - A model trained on a larger (possibly private) dataset

    Reference: Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
    """

    def __init__(
        self,
        teacher: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_distillation: bool = False,
        feature_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.feature_distillation = feature_distillation
        self.feature_layers = feature_layers or []

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Feature projection layers (if feature distillation is enabled)
        self.feature_projectors: nn.ModuleList = nn.ModuleList()

    def setup_feature_projectors(
        self,
        student_dims: List[int],
        teacher_dims: List[int],
    ) -> None:
        """Create projection layers to align student and teacher feature dimensions."""
        self.feature_projectors = nn.ModuleList()
        for s_dim, t_dim in zip(student_dims, teacher_dims):
            if s_dim != t_dim:
                self.feature_projectors.append(
                    nn.Sequential(
                        nn.Linear(s_dim, t_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(t_dim, t_dim),
                    )
                )
            else:
                self.feature_projectors.append(nn.Identity())

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between soft teacher and student distributions."""
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        return kl_loss * (T * T)

    def feature_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute feature-level distillation loss (L2 on projected features)."""
        total_loss = torch.tensor(0.0, device=student_features[0].device)

        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            if i >= len(self.feature_projectors):
                break

            # Pool spatial dimensions if needed
            if s_feat.dim() == 4:
                s_feat = F.adaptive_avg_pool2d(s_feat, 1).flatten(1)
            if t_feat.dim() == 4:
                t_feat = F.adaptive_avg_pool2d(t_feat, 1).flatten(1)

            s_proj = self.feature_projectors[i](s_feat)
            total_loss = total_loss + F.mse_loss(s_proj, t_feat.detach())

        return total_loss / max(len(student_features), 1)

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        student_features: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined distillation and task loss.

        Returns a dict of loss components for logging.
        """
        # Task loss (hard labels)
        task_loss = F.cross_entropy(student_logits, labels)

        # Teacher forward pass
        with torch.no_grad():
            if images is not None:
                teacher_output = self.teacher(images, return_features=True)
                if isinstance(teacher_output, tuple):
                    teacher_logits, teacher_features = teacher_output
                else:
                    teacher_logits = teacher_output
                    teacher_features = None
            else:
                teacher_logits = self.teacher(student_logits)  # fallback
                teacher_features = None

        # Distillation loss (soft labels)
        kd_loss = self.distillation_loss(student_logits, teacher_logits)

        # Combined loss
        total_loss = (1.0 - self.alpha) * task_loss + self.alpha * kd_loss

        losses = {
            "task_loss": task_loss,
            "kd_loss": kd_loss,
            "total_loss": total_loss,
        }

        # Feature distillation (optional)
        if (
            self.feature_distillation
            and student_features is not None
            and teacher_features is not None
        ):
            feat_loss = self.feature_loss(student_features, teacher_features)
            total_loss = total_loss + 0.1 * feat_loss
            losses["feature_loss"] = feat_loss
            losses["total_loss"] = total_loss

        return losses


# ---------------------------------------------------------------------------
# Progressive Training Pipeline
# ---------------------------------------------------------------------------

class ProgressiveTrainingPipeline:
    """Orchestrates all progressive training strategies.

    Coordinates resolution scaling, gradual unfreezing, layer-wise LR decay,
    and knowledge distillation into a unified training pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_groups: List[List[nn.Parameter]],
        base_lr: float = 1e-3,
        lr_decay_factor: float = 0.65,
        resolution_schedule: Optional[ResolutionSchedule] = None,
        warmup_epochs: int = 3,
        unfreeze_every: int = 5,
        teacher: Optional[nn.Module] = None,
        distill_temperature: float = 4.0,
        distill_alpha: float = 0.7,
    ) -> None:
        self.model = model
        self.layer_groups = layer_groups

        # Layer-wise LR
        self.lr_scheduler = LayerWiseLRScheduler(
            layer_groups, base_lr=base_lr, decay_factor=lr_decay_factor
        )

        # Resolution scaling
        self.resolution_trainer: Optional[ProgressiveResolutionTrainer] = None
        if resolution_schedule is not None:
            self.resolution_trainer = ProgressiveResolutionTrainer(
                schedule=resolution_schedule, base_batch_size=32
            )

        # Gradual unfreezing (set up after optimizer is created)
        self.warmup_epochs = warmup_epochs
        self.unfreeze_every = unfreeze_every
        self.unfreezing_scheduler: Optional[GradualUnfreezingScheduler] = None

        # Knowledge distillation
        self.distillation: Optional[KnowledgeDistillation] = None
        if teacher is not None:
            self.distillation = KnowledgeDistillation(
                teacher, temperature=distill_temperature, alpha=distill_alpha
            )

    def setup_optimizer(self) -> Optimizer:
        """Create optimizer with layer-wise learning rates."""
        param_groups = self.lr_scheduler.get_param_groups()
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        self.unfreezing_scheduler = GradualUnfreezingScheduler(
            model=self.model,
            layer_groups=self.layer_groups,
            optimizer=optimizer,
            warmup_epochs=self.warmup_epochs,
            unfreeze_every=self.unfreeze_every,
        )
        return optimizer

    def on_epoch_start(self, epoch: int) -> Dict[str, Any]:
        """Called at the start of each epoch. Returns epoch config."""
        config: Dict[str, Any] = {"epoch": epoch}

        # Unfreezing
        if self.unfreezing_scheduler is not None:
            changed = self.unfreezing_scheduler.step(epoch)
            config["unfreezing_changed"] = changed
            config["unfrozen_from"] = self.unfreezing_scheduler.unfrozen_from

        # Resolution
        if self.resolution_trainer is not None:
            resolution, batch_size = self.resolution_trainer.get_resolution_and_batch_size(epoch)
            config["resolution"] = resolution
            config["batch_size"] = batch_size

        return config

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        student_features: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss, incorporating distillation if available."""
        if self.distillation is not None:
            return self.distillation(
                student_logits, labels, images=images,
                student_features=student_features,
            )
        else:
            loss = F.cross_entropy(student_logits, labels)
            return {"total_loss": loss, "task_loss": loss}

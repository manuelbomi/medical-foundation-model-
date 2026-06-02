"""
Domain Adaptation Strategies for Medical Imaging.

Implements multiple approaches to bridge the domain gap between natural-image
pretrained models and clinical imaging data:
- Progressive fine-tuning with gradual unfreezing
- Domain-Adversarial Neural Networks (DANN) with gradient reversal
- Maximum Mean Discrepancy (MMD) alignment
- Feature alignment via correlation matching
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for DANN)
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """Reverse gradients during backward pass, scaled by lambda.

    This is the core mechanism behind Domain-Adversarial Neural Networks:
    the feature extractor is trained to *fool* the domain classifier by
    reversing the gradient signal flowing from the domain head.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_val * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_val: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to a tensor."""
    return GradientReversalFunction.apply(x, lambda_val)


# ---------------------------------------------------------------------------
# Domain Classifier
# ---------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """Binary domain classifier head for adversarial training.

    Takes pooled feature representations and predicts whether the input
    comes from the source (natural images) or target (medical) domain.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 1024, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, lambda_val: float = 1.0) -> torch.Tensor:
        reversed_features = gradient_reversal(features, lambda_val)
        return self.net(reversed_features)


# ---------------------------------------------------------------------------
# MMD Loss
# ---------------------------------------------------------------------------

class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss with multiple Gaussian kernels.

    Measures the distance between two distributions in a Reproducing Kernel
    Hilbert Space (RKHS). Uses a mixture of Gaussian kernels with different
    bandwidths for robustness to scale.

    Reference: Gretton et al., "A Kernel Two-Sample Test", JMLR 2012
    """

    def __init__(self, kernel_bandwidths: Optional[List[float]] = None) -> None:
        super().__init__()
        if kernel_bandwidths is None:
            kernel_bandwidths = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.kernel_bandwidths = kernel_bandwidths

    def _gaussian_kernel_matrix(
        self, x: torch.Tensor, y: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        """Compute Gaussian kernel matrix between x and y."""
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        y_sq = (y ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_sq + y_sq.T - 2.0 * x @ y.T
        return torch.exp(-dist_sq / (2.0 * bandwidth))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MMD^2 between source and target feature distributions."""
        batch_size = min(source.size(0), target.size(0))
        source = source[:batch_size]
        target = target[:batch_size]

        mmd_loss = torch.tensor(0.0, device=source.device)
        for bw in self.kernel_bandwidths:
            k_ss = self._gaussian_kernel_matrix(source, source, bw)
            k_tt = self._gaussian_kernel_matrix(target, target, bw)
            k_st = self._gaussian_kernel_matrix(source, target, bw)
            mmd_loss = mmd_loss + k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()

        return mmd_loss / len(self.kernel_bandwidths)


# ---------------------------------------------------------------------------
# CORAL Loss (Correlation Alignment)
# ---------------------------------------------------------------------------

class CORALLoss(nn.Module):
    """Correlation Alignment loss.

    Aligns second-order statistics (covariance) of source and target
    feature distributions. Computationally lighter than MMD while still
    effective for reducing domain shift.

    Reference: Sun & Saenko, "Deep CORAL", ECCV Workshops 2016
    """

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = source.size(1)
        n_s = source.size(0)
        n_t = target.size(0)

        source_centered = source - source.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)

        cov_source = (source_centered.T @ source_centered) / max(n_s - 1, 1)
        cov_target = (target_centered.T @ target_centered) / max(n_t - 1, 1)

        loss = (cov_source - cov_target).pow(2).sum() / (4 * d * d)
        return loss


# ---------------------------------------------------------------------------
# Adaptation Strategy Interface
# ---------------------------------------------------------------------------

@dataclass
class AdaptationState:
    """Tracks the state of domain adaptation across training."""
    current_epoch: int = 0
    total_epochs: int = 100
    lambda_schedule: str = "linear"  # linear, exp, step
    max_lambda: float = 1.0
    unfrozen_layers: int = 0

    @property
    def progress(self) -> float:
        return self.current_epoch / max(self.total_epochs, 1)

    @property
    def current_lambda(self) -> float:
        """Compute current GRL lambda based on schedule."""
        p = self.progress
        if self.lambda_schedule == "linear":
            return self.max_lambda * p
        elif self.lambda_schedule == "exp":
            return self.max_lambda * (2.0 / (1.0 + (-10.0 * p).__exp__()) - 1.0)
        elif self.lambda_schedule == "step":
            if p < 0.33:
                return self.max_lambda * 0.1
            elif p < 0.66:
                return self.max_lambda * 0.5
            return self.max_lambda
        return self.max_lambda * p


class DomainAdaptationStrategy(ABC):
    """Base class for domain adaptation strategies."""

    @abstractmethod
    def compute_adaptation_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        state: AdaptationState,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def get_extra_modules(self) -> Dict[str, nn.Module]:
        """Return additional modules that need to be trained."""
        ...


class DANNStrategy(DomainAdaptationStrategy):
    """Domain-Adversarial Neural Network strategy.

    Trains a domain classifier with gradient reversal to learn
    domain-invariant feature representations. The GRL lambda is
    scheduled to increase over training for stable convergence.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 1024) -> None:
        self.domain_classifier = DomainClassifier(feature_dim, hidden_dim)

    def compute_adaptation_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        state: AdaptationState,
    ) -> torch.Tensor:
        lambda_val = state.current_lambda

        # Pool features if they have spatial dimensions
        if source_features.dim() == 4:
            source_features = F.adaptive_avg_pool2d(source_features, 1).flatten(1)
        if target_features.dim() == 4:
            target_features = F.adaptive_avg_pool2d(target_features, 1).flatten(1)

        combined = torch.cat([source_features, target_features], dim=0)
        domain_preds = self.domain_classifier(combined, lambda_val)

        # Labels: 0 = source, 1 = target
        labels = torch.cat([
            torch.zeros(source_features.size(0)),
            torch.ones(target_features.size(0)),
        ]).to(domain_preds.device).unsqueeze(1)

        return F.binary_cross_entropy_with_logits(domain_preds, labels)

    def get_extra_modules(self) -> Dict[str, nn.Module]:
        return {"domain_classifier": self.domain_classifier}


class MMDStrategy(DomainAdaptationStrategy):
    """MMD-based domain adaptation.

    Minimizes the Maximum Mean Discrepancy between source and target
    feature distributions. Can be applied at multiple layers for
    stronger alignment.
    """

    def __init__(
        self,
        kernel_bandwidths: Optional[List[float]] = None,
        loss_weight: float = 1.0,
    ) -> None:
        self.mmd_loss = MMDLoss(kernel_bandwidths)
        self.loss_weight = loss_weight

    def compute_adaptation_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        state: AdaptationState,
    ) -> torch.Tensor:
        if source_features.dim() == 4:
            source_features = F.adaptive_avg_pool2d(source_features, 1).flatten(1)
        if target_features.dim() == 4:
            target_features = F.adaptive_avg_pool2d(target_features, 1).flatten(1)

        weight = self.loss_weight * state.current_lambda
        return weight * self.mmd_loss(source_features, target_features)

    def get_extra_modules(self) -> Dict[str, nn.Module]:
        return {}


class CombinedStrategy(DomainAdaptationStrategy):
    """Combine multiple adaptation strategies with weighted contributions."""

    def __init__(
        self,
        strategies: List[Tuple[DomainAdaptationStrategy, float]],
    ) -> None:
        self.strategies = strategies

    def compute_adaptation_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        state: AdaptationState,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=source_features.device)
        for strategy, weight in self.strategies:
            total_loss = total_loss + weight * strategy.compute_adaptation_loss(
                source_features, target_features, state
            )
        return total_loss

    def get_extra_modules(self) -> Dict[str, nn.Module]:
        modules = {}
        for i, (strategy, _) in enumerate(self.strategies):
            for name, module in strategy.get_extra_modules().items():
                modules[f"strategy_{i}_{name}"] = module
        return modules


# ---------------------------------------------------------------------------
# Progressive Fine-Tuning Controller
# ---------------------------------------------------------------------------

class ProgressiveUnfreezer:
    """Gradually unfreeze backbone layers during fine-tuning.

    Starts with all backbone layers frozen (linear probing) and progressively
    unfreezes from the top (task-specific) layers to the bottom (generic features).
    This preserves low-level features learned on ImageNet while allowing high-level
    adaptation to the medical domain.
    """

    def __init__(
        self,
        layer_groups: List[List[nn.Parameter]],
        warmup_epochs: int = 5,
        unfreeze_every: int = 3,
        initial_frozen: int = -1,
    ) -> None:
        self.layer_groups = layer_groups
        self.warmup_epochs = warmup_epochs
        self.unfreeze_every = unfreeze_every
        self.n_groups = len(layer_groups)

        # Initially freeze all groups except the last (classifier head)
        if initial_frozen < 0:
            initial_frozen = self.n_groups - 1
        self.frozen_up_to = min(initial_frozen, self.n_groups - 1)

        self._apply_freeze()

    def _apply_freeze(self) -> None:
        """Freeze groups below the threshold, unfreeze above."""
        for i, group in enumerate(self.layer_groups):
            frozen = i < self.frozen_up_to
            for param in group:
                param.requires_grad = not frozen

        logger.info(
            f"Layers frozen: groups 0..{self.frozen_up_to - 1} "
            f"(of {self.n_groups} total), {self._count_trainable()} trainable params"
        )

    def _count_trainable(self) -> int:
        count = 0
        for group in self.layer_groups:
            for p in group:
                if p.requires_grad:
                    count += p.numel()
        return count

    def step(self, epoch: int) -> bool:
        """Check if we should unfreeze another layer group. Returns True if changed."""
        if epoch < self.warmup_epochs:
            return False
        if self.frozen_up_to <= 0:
            return False

        adjusted_epoch = epoch - self.warmup_epochs
        if adjusted_epoch > 0 and adjusted_epoch % self.unfreeze_every == 0:
            self.frozen_up_to -= 1
            self._apply_freeze()
            logger.info(f"Epoch {epoch}: unfroze group {self.frozen_up_to}")
            return True
        return False

    @property
    def fully_unfrozen(self) -> bool:
        return self.frozen_up_to <= 0


# ---------------------------------------------------------------------------
# DomainAdapter: Orchestrator
# ---------------------------------------------------------------------------

class DomainAdapter:
    """High-level orchestrator for domain adaptation.

    Combines a backbone model with an adaptation strategy and progressive
    unfreezing schedule to manage the full adaptation pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: DomainAdaptationStrategy,
        layer_groups: Optional[List[List[nn.Parameter]]] = None,
        warmup_epochs: int = 5,
        unfreeze_every: int = 3,
        total_epochs: int = 100,
        lambda_schedule: str = "exp",
        max_lambda: float = 1.0,
    ) -> None:
        self.model = model
        self.strategy = strategy

        self.state = AdaptationState(
            total_epochs=total_epochs,
            lambda_schedule=lambda_schedule,
            max_lambda=max_lambda,
        )

        self.unfreezer: Optional[ProgressiveUnfreezer] = None
        if layer_groups is not None:
            self.unfreezer = ProgressiveUnfreezer(
                layer_groups,
                warmup_epochs=warmup_epochs,
                unfreeze_every=unfreeze_every,
            )

    def on_epoch_start(self, epoch: int) -> None:
        """Update adaptation state at the start of each epoch."""
        self.state.current_epoch = epoch
        if self.unfreezer is not None:
            self.unfreezer.step(epoch)

    def compute_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the domain adaptation loss for the current training state."""
        return self.strategy.compute_adaptation_loss(
            source_features, target_features, self.state
        )

    def get_trainable_params(self) -> List[Dict[str, Any]]:
        """Return optimizer parameter groups including adaptation modules."""
        params = [{"params": p} for p in self.model.parameters() if p.requires_grad]
        for name, module in self.strategy.get_extra_modules().items():
            params.append({
                "params": module.parameters(),
                "lr_scale": 10.0,  # Train domain head faster
                "name": name,
            })
        return params

"""
Few-Shot Learning for Rare Pathology Detection in Medical Imaging.

Implements meta-learning approaches for scenarios where only a handful
of annotated examples exist for a given pathology. This is common in
medical imaging: rare diseases, unusual presentations, and newly defined
diagnostic categories all suffer from extreme label scarcity.

Methods:
- Prototypical Networks (class-prototype matching in embedding space)
- MAML (Model-Agnostic Meta-Learning) wrapper
- Support set augmentation strategies
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


@dataclass
class EpisodeConfig:
    """Configuration for few-shot episode sampling."""
    n_way: int = 5       # Number of classes per episode
    n_support: int = 5   # Support (training) examples per class
    n_query: int = 15    # Query (evaluation) examples per class
    n_episodes: int = 100  # Episodes per epoch


class EpisodeSampler:
    """Sample few-shot episodes from a dataset.

    Each episode consists of:
    - Support set: n_way * n_support labeled examples
    - Query set: n_way * n_query labeled examples for evaluation

    Classes are sampled randomly per episode, simulating the meta-learning
    task distribution.
    """

    def __init__(
        self,
        labels: np.ndarray,
        config: EpisodeConfig,
    ) -> None:
        self.labels = labels
        self.config = config
        self.n_samples = len(labels)

        # Build per-class index lookup
        self.class_indices: Dict[int, np.ndarray] = {}
        for cls in np.unique(labels):
            self.class_indices[int(cls)] = np.where(labels == cls)[0]

        # Filter classes with enough samples
        min_required = config.n_support + config.n_query
        self.available_classes = [
            cls for cls, indices in self.class_indices.items()
            if len(indices) >= min_required
        ]

        if len(self.available_classes) < config.n_way:
            logger.warning(
                f"Only {len(self.available_classes)} classes have >= {min_required} samples, "
                f"but n_way={config.n_way}. Some episodes may reuse classes."
            )

    def sample_episode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a single episode.

        Returns:
            support_indices: Indices into the dataset for support set
            support_labels: Remapped labels (0 to n_way-1)
            query_indices: Indices for query set
            query_labels: Remapped labels
        """
        # Select n_way classes
        n_way = min(self.config.n_way, len(self.available_classes))
        selected_classes = random.sample(self.available_classes, n_way)

        support_indices = []
        support_labels = []
        query_indices = []
        query_labels = []

        for new_label, cls in enumerate(selected_classes):
            cls_indices = self.class_indices[cls]
            sampled = np.random.choice(
                cls_indices,
                size=self.config.n_support + self.config.n_query,
                replace=False,
            )
            support = sampled[: self.config.n_support]
            query = sampled[self.config.n_support :]

            support_indices.extend(support)
            support_labels.extend([new_label] * self.config.n_support)
            query_indices.extend(query)
            query_labels.extend([new_label] * self.config.n_query)

        return (
            np.array(support_indices),
            np.array(support_labels),
            np.array(query_indices),
            np.array(query_labels),
        )


# ---------------------------------------------------------------------------
# Prototypical Networks
# ---------------------------------------------------------------------------

class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot classification.

    Computes class prototypes as the mean embedding of support examples,
    then classifies query examples by distance to prototypes.

    In medical imaging, this approach is attractive because:
    - No fine-tuning needed at test time (just compute prototypes)
    - Naturally handles any number of classes
    - The learned embedding can be inspected for clinical interpretability

    Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        distance: str = "euclidean",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.distance = distance
        self.temperature = temperature

        # Optional learnable projection to a metric-friendly space
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project embeddings."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.projection(features)

    def compute_prototypes(
        self, support_embeddings: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute class prototypes as mean of support embeddings.

        Args:
            support_embeddings: (n_support_total, feature_dim)
            support_labels: (n_support_total,) with values in [0, n_way)

        Returns:
            Prototypes tensor (n_way, feature_dim)
        """
        classes = torch.unique(support_labels)
        prototypes = []
        for cls in classes:
            mask = support_labels == cls
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def compute_distances(
        self, query_embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances from queries to prototypes.

        Args:
            query_embeddings: (n_query, feature_dim)
            prototypes: (n_way, feature_dim)

        Returns:
            Negative distances (n_query, n_way) -- higher = closer
        """
        if self.distance == "euclidean":
            # (n_query, 1, dim) - (1, n_way, dim)
            diffs = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = -(diffs ** 2).sum(dim=-1)
        elif self.distance == "cosine":
            query_norm = F.normalize(query_embeddings, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            distances = query_norm @ proto_norm.T
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        return distances / self.temperature

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Classify query images given support set.

        Returns logits (n_query, n_way) based on distances to prototypes.
        """
        support_embeddings = self.embed(support_images)
        query_embeddings = self.embed(query_images)
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        logits = self.compute_distances(query_embeddings, prototypes)
        return logits

    def episode_loss(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Compute loss and accuracy for a single episode."""
        logits = self(support_images, support_labels, query_images)
        loss = F.cross_entropy(logits, query_labels)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean().item()
        return loss, accuracy


# ---------------------------------------------------------------------------
# MAML Wrapper
# ---------------------------------------------------------------------------

class MAMLWrapper(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) wrapper.

    Learns an initialization that can be rapidly adapted to new tasks
    (classes) with just a few gradient steps on a small support set.

    MAML is particularly powerful for medical imaging because:
    - The learned initialization encodes transferable medical features
    - Inner-loop adaptation handles task-specific (pathology-specific) tuning
    - Naturally handles the "new rare pathology" scenario

    Reference: Finn et al., "Model-Agnostic Meta-Learning", ICML 2017
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

    def _inner_loop(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run inner loop adaptation on the support set.

        Creates a temporary copy of parameters and performs gradient descent
        on the support loss.
        """
        adapted_params = {k: v.clone() for k, v in params.items()}

        for step in range(self.inner_steps):
            logits = self._forward_with_params(support_images, adapted_params)
            loss = F.cross_entropy(logits, support_labels)

            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            adapted_params = {
                k: v - self.inner_lr * (g if g is not None else torch.zeros_like(v))
                for (k, v), g in zip(adapted_params.items(), grads)
            }

        return adapted_params

    def _forward_with_params(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using the given parameters (for inner-loop differentiation).

        Uses functional form to allow gradient computation through the
        adaptation process.
        """
        # For simplicity, use the model's forward and temporarily swap parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            if name in params:
                original_params[name] = param.data.clone()
                param.data = params[name]

        output = self.model(x)

        # Restore original parameters
        for name, data in original_params.items():
            dict(self.model.named_parameters())[name].data = data

        return output

    def meta_train_step(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Perform one meta-training step (outer loop).

        1. Adapt parameters on support set (inner loop)
        2. Evaluate adapted parameters on query set
        3. Backprop through the adaptation to update the initialization
        """
        params = dict(self.model.named_parameters())
        adapted_params = self._inner_loop(support_images, support_labels, params)

        # Evaluate on query set with adapted parameters
        query_logits = self._forward_with_params(query_images, adapted_params)
        meta_loss = F.cross_entropy(query_logits, query_labels)

        preds = query_logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean().item()

        return meta_loss, accuracy

    @torch.no_grad()
    def adapt_and_evaluate(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt to a new task and classify query images (inference)."""
        params = dict(self.model.named_parameters())
        adapted_params = self._inner_loop(support_images, support_labels, params)
        logits = self._forward_with_params(query_images, adapted_params)
        return logits


# ---------------------------------------------------------------------------
# Support Set Augmentation
# ---------------------------------------------------------------------------

class SupportSetAugmentor:
    """Augmentation strategies specifically designed for few-shot support sets.

    When you only have 1-5 examples per class, augmentation is critical.
    Medical image augmentations must be domain-appropriate:
    - No color jitter (grayscale images)
    - Careful with flips (laterality matters in some modalities)
    - Elastic deformations (simulate anatomical variation)
    """

    def __init__(
        self,
        augmentation_factor: int = 10,
        include_elastic: bool = True,
        include_intensity: bool = True,
        include_geometric: bool = True,
    ) -> None:
        self.augmentation_factor = augmentation_factor
        self.include_elastic = include_elastic
        self.include_intensity = include_intensity
        self.include_geometric = include_geometric

    def _elastic_deformation(
        self, image: torch.Tensor, alpha: float = 50.0, sigma: float = 5.0
    ) -> torch.Tensor:
        """Apply random elastic deformation to simulate anatomical variation."""
        B, C, H, W = image.shape
        dx = torch.randn(B, 1, H, W, device=image.device) * alpha / H
        dy = torch.randn(B, 1, H, W, device=image.device) * alpha / W

        # Smooth displacement fields
        kernel_size = int(6 * sigma + 1) | 1
        if kernel_size > 1:
            padding = kernel_size // 2
            kernel_1d = torch.exp(
                -torch.arange(kernel_size, dtype=torch.float32, device=image.device).sub(padding).pow(2)
                / (2 * sigma ** 2)
            )
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

            dx = F.conv2d(dx, kernel_2d, padding=padding)
            dy = F.conv2d(dy, kernel_2d, padding=padding)

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        displacement = torch.cat([dy, dx], dim=1).permute(0, 2, 3, 1)
        grid = grid + displacement

        return F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

    def _intensity_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random intensity transformations."""
        B = image.shape[0]

        # Random brightness
        brightness = 1.0 + 0.2 * (torch.rand(B, 1, 1, 1, device=image.device) - 0.5)
        image = image * brightness

        # Random contrast
        mean_val = image.mean(dim=(-2, -1), keepdim=True)
        contrast = 1.0 + 0.3 * (torch.rand(B, 1, 1, 1, device=image.device) - 0.5)
        image = mean_val + contrast * (image - mean_val)

        # Random Gaussian noise
        noise_std = 0.02 * torch.rand(B, 1, 1, 1, device=image.device)
        image = image + torch.randn_like(image) * noise_std

        return image.clamp(0, 1)

    def _geometric_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random geometric transformations (rotation, scale)."""
        B, C, H, W = image.shape

        # Random rotation (small angles for medical images)
        angle = (torch.rand(B, device=image.device) - 0.5) * 20.0  # +/- 10 degrees
        angle_rad = angle * (3.14159 / 180.0)

        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Random scale
        scale = 0.9 + 0.2 * torch.rand(B, device=image.device)

        # Build affine matrices
        theta = torch.zeros(B, 2, 3, device=image.device)
        theta[:, 0, 0] = cos_a * scale
        theta[:, 0, 1] = -sin_a * scale
        theta[:, 1, 0] = sin_a * scale
        theta[:, 1, 1] = cos_a * scale

        grid = F.affine_grid(theta, image.shape, align_corners=True)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

    def augment_support_set(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment support set images to create a larger effective support set.

        Args:
            images: Support images (n_support, C, H, W)
            labels: Support labels (n_support,)

        Returns:
            Augmented images and labels (n_support * augmentation_factor, C, H, W)
        """
        all_images = [images]
        all_labels = [labels]

        for _ in range(self.augmentation_factor - 1):
            aug_images = images.clone()

            if self.include_geometric:
                aug_images = self._geometric_augmentation(aug_images)
            if self.include_intensity:
                aug_images = self._intensity_augmentation(aug_images)
            if self.include_elastic:
                aug_images = self._elastic_deformation(aug_images)

            all_images.append(aug_images)
            all_labels.append(labels.clone())

        return torch.cat(all_images, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Few-Shot Evaluator
# ---------------------------------------------------------------------------

class FewShotEvaluator:
    """Evaluate few-shot learning performance across multiple episodes."""

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        config: EpisodeConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device

        # Extract all labels
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            labels.append(item[1] if isinstance(item, (list, tuple)) else 0)
        self.labels = np.array(labels)

        self.sampler = EpisodeSampler(self.labels, config)

    @torch.no_grad()
    def evaluate(self, n_episodes: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation over multiple episodes and report statistics."""
        self.model.eval()
        n_episodes = n_episodes or self.config.n_episodes

        accuracies = []
        for ep in range(n_episodes):
            support_idx, support_labels, query_idx, query_labels = self.sampler.sample_episode()

            support_images = torch.stack([self.dataset[i][0] for i in support_idx]).to(self.device)
            query_images = torch.stack([self.dataset[i][0] for i in query_idx]).to(self.device)
            support_labels_t = torch.tensor(support_labels, device=self.device)
            query_labels_t = torch.tensor(query_labels, device=self.device)

            if isinstance(self.model, PrototypicalNetwork):
                logits = self.model(support_images, support_labels_t, query_images)
            elif isinstance(self.model, MAMLWrapper):
                logits = self.model.adapt_and_evaluate(
                    support_images, support_labels_t, query_images
                )
            else:
                logits = self.model(query_images)

            preds = logits.argmax(dim=-1)
            acc = (preds == query_labels_t).float().mean().item()
            accuracies.append(acc)

        accs = np.array(accuracies)
        ci_95 = 1.96 * accs.std() / np.sqrt(len(accs))

        results = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "ci_95": float(ci_95),
            "min_accuracy": float(accs.min()),
            "max_accuracy": float(accs.max()),
            "n_episodes": n_episodes,
        }

        logger.info(
            f"Few-shot evaluation ({n_episodes} episodes): "
            f"{results['mean_accuracy']:.4f} +/- {results['ci_95']:.4f}"
        )
        return results

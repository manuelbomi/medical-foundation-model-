"""
Difficulty Scoring for Medical Image Training Samples.

Assigns difficulty scores to individual training examples using multiple
complementary signals. In medical imaging, difficulty correlates with
clinical subtlety -- hard cases are often those with ambiguous findings,
poor image quality, or rare presentations.

Scoring methods:
- Loss-based: samples with higher training loss are harder
- Prediction uncertainty: entropy or variance of model predictions
- Data complexity: image quality metrics, finding subtlety
- Radiologist agreement: inter-reader variability as difficulty proxy
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class DifficultyReport:
    """Report containing difficulty scores and metadata for a dataset."""
    scores: np.ndarray
    method: str
    score_stats: Dict[str, float]
    per_class_stats: Optional[Dict[int, Dict[str, float]]] = None

    def summary(self) -> str:
        lines = [
            f"Difficulty Report ({self.method})",
            f"  N samples: {len(self.scores)}",
            f"  Mean: {self.score_stats['mean']:.4f}",
            f"  Std:  {self.score_stats['std']:.4f}",
            f"  Min:  {self.score_stats['min']:.4f}",
            f"  Max:  {self.score_stats['max']:.4f}",
        ]
        if self.per_class_stats:
            for cls, stats in self.per_class_stats.items():
                lines.append(f"  Class {cls}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return "\n".join(lines)


class DifficultyScorer(ABC):
    """Base class for difficulty scoring methods."""

    @abstractmethod
    def score(self, **kwargs: Any) -> DifficultyReport:
        """Compute difficulty scores for all samples."""
        ...

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max scaling."""
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-8:
            return np.full_like(scores, 0.5)
        return (scores - s_min) / (s_max - s_min)

    @staticmethod
    def compute_stats(
        scores: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, float], Optional[Dict[int, Dict[str, float]]]]:
        """Compute summary statistics for difficulty scores."""
        stats = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "median": float(np.median(scores)),
        }

        per_class = None
        if labels is not None:
            per_class = {}
            for cls in np.unique(labels):
                mask = labels == cls
                per_class[int(cls)] = {
                    "mean": float(scores[mask].mean()),
                    "std": float(scores[mask].std()),
                    "count": int(mask.sum()),
                }

        return stats, per_class


class LossBasedScorer(DifficultyScorer):
    """Score difficulty based on training loss.

    Samples that consistently have high loss across multiple epochs are
    considered harder. Uses exponential moving average to smooth noisy
    per-sample losses.

    This is the most direct difficulty measure: the model tells us which
    samples it finds hard by failing to fit them.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        n_epochs: int = 5,
        batch_size: int = 32,
        momentum: float = 0.9,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.momentum = momentum

    @torch.no_grad()
    def score(self, **kwargs: Any) -> DifficultyReport:
        """Compute loss-based difficulty scores via multiple forward passes."""
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        n_samples = len(self.dataset)
        cumulative_losses = np.zeros(n_samples)
        all_labels = np.zeros(n_samples, dtype=np.int64)

        for epoch in range(self.n_epochs):
            idx_offset = 0
            for batch in loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(images)

                # Per-sample cross-entropy loss
                per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
                losses_np = per_sample_loss.cpu().numpy()
                bs = len(losses_np)

                if epoch == 0:
                    cumulative_losses[idx_offset : idx_offset + bs] = losses_np
                    all_labels[idx_offset : idx_offset + bs] = labels.cpu().numpy()
                else:
                    cumulative_losses[idx_offset : idx_offset + bs] = (
                        self.momentum * cumulative_losses[idx_offset : idx_offset + bs]
                        + (1 - self.momentum) * losses_np
                    )
                idx_offset += bs

        scores = self.normalize_scores(cumulative_losses)
        stats, per_class = self.compute_stats(scores, all_labels)

        return DifficultyReport(
            scores=scores, method="loss_based", score_stats=stats, per_class_stats=per_class
        )


class UncertaintyScorer(DifficultyScorer):
    """Score difficulty using prediction uncertainty (MC Dropout or ensemble).

    Runs multiple stochastic forward passes with dropout enabled and
    measures prediction variance/entropy. High uncertainty indicates the
    model is unsure -- a sign of a difficult or ambiguous sample.

    This captures a different notion of difficulty than loss: a sample can
    have moderate loss but very high uncertainty (conflicting signals).
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        n_forward_passes: int = 20,
        batch_size: int = 32,
        uncertainty_metric: str = "entropy",  # entropy, variance, mutual_info
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.n_forward_passes = n_forward_passes
        self.batch_size = batch_size
        self.uncertainty_metric = uncertainty_metric

    def _enable_mc_dropout(self) -> None:
        """Enable dropout layers during inference for MC Dropout."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def score(self, **kwargs: Any) -> DifficultyReport:
        """Compute uncertainty-based difficulty scores."""
        self.model.eval()
        self._enable_mc_dropout()

        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        n_samples = len(self.dataset)
        all_probs: List[np.ndarray] = []

        for _ in range(self.n_forward_passes):
            pass_probs = []
            for batch in loader:
                images = batch[0].to(self.device)
                logits = self.model(images)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                pass_probs.append(probs)
            all_probs.append(np.concatenate(pass_probs, axis=0))

        # Stack: (n_passes, n_samples, n_classes)
        stacked = np.stack(all_probs, axis=0)
        mean_probs = stacked.mean(axis=0)  # (n_samples, n_classes)

        if self.uncertainty_metric == "entropy":
            # Predictive entropy
            scores_raw = -np.sum(
                mean_probs * np.log(mean_probs + 1e-10), axis=1
            )
        elif self.uncertainty_metric == "variance":
            # Mean variance across classes
            scores_raw = stacked.var(axis=0).mean(axis=1)
        elif self.uncertainty_metric == "mutual_info":
            # Mutual information = predictive entropy - expected entropy
            pred_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
            expected_entropy = -np.mean(
                np.sum(stacked * np.log(stacked + 1e-10), axis=2), axis=0
            )
            scores_raw = pred_entropy - expected_entropy
        else:
            raise ValueError(f"Unknown metric: {self.uncertainty_metric}")

        scores = self.normalize_scores(scores_raw)

        # Collect labels
        all_labels = []
        for batch in DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False):
            all_labels.append(batch[1].numpy())
        labels = np.concatenate(all_labels)

        stats, per_class = self.compute_stats(scores, labels)

        return DifficultyReport(
            scores=scores,
            method=f"uncertainty_{self.uncertainty_metric}",
            score_stats=stats,
            per_class_stats=per_class,
        )


class DataComplexityScorer(DifficultyScorer):
    """Score difficulty based on intrinsic data complexity.

    Uses image-level features as proxies for difficulty:
    - Image quality (noise level, contrast, sharpness)
    - Finding subtlety (lesion size, contrast with background)
    - Anatomical complexity (tissue density, overlapping structures)

    This scorer works without model predictions, making it suitable for
    cold-start curriculum design before any training has occurred.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.metadata = metadata or {}

    def _compute_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using median absolute deviation of Laplacian."""
        # Approximate Laplacian using finite differences
        if image.ndim == 3:
            image = image.mean(axis=0)
        laplacian = (
            np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0)
            + np.roll(image, 1, axis=1) + np.roll(image, -1, axis=1)
            - 4 * image
        )
        # Robust noise estimate
        sigma = np.median(np.abs(laplacian)) * 1.4826
        return float(sigma)

    def _compute_contrast(self, image: np.ndarray) -> float:
        """Compute Michelson contrast."""
        if image.ndim == 3:
            image = image.mean(axis=0)
        i_max = image.max()
        i_min = image.min()
        if i_max + i_min < 1e-8:
            return 0.0
        return float((i_max - i_min) / (i_max + i_min))

    def _compute_sharpness(self, image: np.ndarray) -> float:
        """Compute image sharpness as variance of Laplacian."""
        if image.ndim == 3:
            image = image.mean(axis=0)
        laplacian = (
            np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0)
            + np.roll(image, 1, axis=1) + np.roll(image, -1, axis=1)
            - 4 * image
        )
        return float(np.var(laplacian))

    def _compute_tissue_density(self, image: np.ndarray) -> float:
        """Estimate breast tissue density as fraction of bright pixels.

        Higher density makes lesion detection harder (mammography-specific).
        """
        if image.ndim == 3:
            image = image.mean(axis=0)
        threshold = np.percentile(image, 50)
        return float((image > threshold).mean())

    def score(self, **kwargs: Any) -> DifficultyReport:
        """Compute data complexity difficulty scores."""
        n = len(self.images)
        noise_scores = np.zeros(n)
        contrast_scores = np.zeros(n)
        sharpness_scores = np.zeros(n)
        density_scores = np.zeros(n)

        for i in range(n):
            img = self.images[i]
            noise_scores[i] = self._compute_noise_level(img)
            contrast_scores[i] = self._compute_contrast(img)
            sharpness_scores[i] = self._compute_sharpness(img)
            density_scores[i] = self._compute_tissue_density(img)

        # Normalize each component
        noise_norm = self.normalize_scores(noise_scores)
        contrast_norm = 1.0 - self.normalize_scores(contrast_scores)  # low contrast = harder
        sharpness_norm = 1.0 - self.normalize_scores(sharpness_scores)  # low sharpness = harder
        density_norm = self.normalize_scores(density_scores)  # high density = harder

        # Combined score (equal weighting)
        combined = (noise_norm + contrast_norm + sharpness_norm + density_norm) / 4.0
        scores = self.normalize_scores(combined)

        stats, per_class = self.compute_stats(scores, self.labels)

        return DifficultyReport(
            scores=scores,
            method="data_complexity",
            score_stats=stats,
            per_class_stats=per_class,
        )


class RadiologistAgreementScorer(DifficultyScorer):
    """Score difficulty using inter-reader agreement as a proxy.

    In medical imaging, samples where radiologists disagree are inherently
    ambiguous and thus harder. This scorer uses agreement metrics from
    multi-reader studies as difficulty labels.

    Agreement can be measured as:
    - Cohen's kappa between reader pairs
    - Fleiss' kappa for multiple readers
    - Simple percentage agreement
    """

    def __init__(
        self,
        reader_annotations: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            reader_annotations: (n_samples, n_readers) array of labels.
                                Use -1 for missing annotations.
            labels: Ground truth labels (majority vote if not provided).
        """
        self.reader_annotations = reader_annotations
        self.n_samples = reader_annotations.shape[0]
        self.n_readers = reader_annotations.shape[1]

        if labels is not None:
            self.labels = labels
        else:
            # Majority vote (ignoring -1)
            self.labels = np.array([
                np.bincount(row[row >= 0]).argmax() if (row >= 0).any() else 0
                for row in reader_annotations
            ])

    def _compute_agreement(self, annotations: np.ndarray) -> float:
        """Compute pairwise agreement for a single sample."""
        valid = annotations[annotations >= 0]
        if len(valid) < 2:
            return 1.0  # Single reader = no disagreement info

        n_valid = len(valid)
        agree_count = 0
        pair_count = 0
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                if valid[i] == valid[j]:
                    agree_count += 1
                pair_count += 1

        return agree_count / max(pair_count, 1)

    def score(self, **kwargs: Any) -> DifficultyReport:
        """Compute difficulty scores based on inter-reader agreement."""
        agreement_scores = np.array([
            self._compute_agreement(self.reader_annotations[i])
            for i in range(self.n_samples)
        ])

        # Invert: low agreement = high difficulty
        raw_difficulty = 1.0 - agreement_scores
        scores = self.normalize_scores(raw_difficulty)

        stats, per_class = self.compute_stats(scores, self.labels)

        return DifficultyReport(
            scores=scores,
            method="radiologist_agreement",
            score_stats=stats,
            per_class_stats=per_class,
        )


class EnsembleDifficultyScorer:
    """Combine multiple difficulty scoring methods into a unified score.

    Different scoring methods capture different aspects of difficulty.
    Combining them provides a more robust and comprehensive difficulty
    estimate.
    """

    def __init__(
        self,
        scorers: List[Tuple[DifficultyScorer, float]],
    ) -> None:
        """
        Args:
            scorers: List of (scorer, weight) tuples.
        """
        self.scorers = scorers

    def score(self, **kwargs: Any) -> DifficultyReport:
        """Compute weighted ensemble of difficulty scores."""
        reports = []
        total_weight = sum(w for _, w in self.scorers)

        combined_scores: Optional[np.ndarray] = None

        for scorer, weight in self.scorers:
            report = scorer.score(**kwargs)
            reports.append(report)
            normalized_weight = weight / total_weight

            if combined_scores is None:
                combined_scores = normalized_weight * report.scores
            else:
                combined_scores += normalized_weight * report.scores

        assert combined_scores is not None
        scores = DifficultyScorer.normalize_scores(combined_scores)

        stats, _ = DifficultyScorer.compute_stats(scores)
        methods = "+".join(r.method for r in reports)

        return DifficultyReport(
            scores=scores,
            method=f"ensemble({methods})",
            score_stats=stats,
        )

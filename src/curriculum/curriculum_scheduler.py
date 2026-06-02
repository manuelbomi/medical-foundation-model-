"""
Curriculum Learning Schedulers for Medical Image Training.

Implements strategies that control the order and pacing of training examples
based on difficulty, enabling smoother optimization landscapes and better
generalization -- especially important in medical imaging where class imbalance
and label noise are prevalent.

Strategies:
- Difficulty-based curriculum (easy to hard)
- Self-paced learning with dynamic thresholds
- Anti-curriculum (hard to easy, for robustness)
- Teacher-student curriculum with mentor model guidance
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


@dataclass
class CurriculumState:
    """Tracks the progression of curriculum learning."""
    epoch: int = 0
    total_epochs: int = 100
    current_difficulty_threshold: float = 0.0
    samples_seen: int = 0
    total_samples: int = 0
    phase: str = "easy"  # easy, medium, hard, all

    @property
    def progress(self) -> float:
        return self.epoch / max(self.total_epochs, 1)


class CurriculumScheduler(ABC):
    """Base class for curriculum learning schedulers.

    A curriculum scheduler determines which subset of training samples
    to present at each epoch and in what order, based on difficulty scores
    assigned to each sample.
    """

    def __init__(self, difficulty_scores: np.ndarray, total_epochs: int = 100) -> None:
        self.difficulty_scores = difficulty_scores
        self.n_samples = len(difficulty_scores)
        self.state = CurriculumState(total_epochs=total_epochs, total_samples=self.n_samples)

        # Sort indices by difficulty (ascending = easy first)
        self.sorted_indices = np.argsort(difficulty_scores)

    @abstractmethod
    def get_sample_indices(self, epoch: int) -> np.ndarray:
        """Return indices of samples to include in training for this epoch."""
        ...

    @abstractmethod
    def get_sample_weights(self, epoch: int) -> np.ndarray:
        """Return per-sample weights for weighted sampling."""
        ...

    def get_sampler(self, epoch: int) -> Sampler:
        """Build a PyTorch Sampler for the current epoch."""
        indices = self.get_sample_indices(epoch)
        weights = self.get_sample_weights(epoch)

        # Subset weights to active indices
        active_weights = weights[indices]
        active_weights = active_weights / active_weights.sum()

        return WeightedRandomSampler(
            weights=active_weights.tolist(),
            num_samples=len(indices),
            replacement=True,
        )

    def step(self, epoch: int) -> None:
        """Update internal state for the new epoch."""
        self.state.epoch = epoch
        self._update_phase()

    def _update_phase(self) -> None:
        p = self.state.progress
        if p < 0.3:
            self.state.phase = "easy"
        elif p < 0.6:
            self.state.phase = "medium"
        elif p < 0.85:
            self.state.phase = "hard"
        else:
            self.state.phase = "all"


class DifficultyBasedCurriculum(CurriculumScheduler):
    """Classic curriculum: start with easy samples, gradually include harder ones.

    The fraction of the dataset included grows linearly (or according to a
    pacing function) from `initial_fraction` to 1.0 over the training run.

    In medical imaging, "easy" samples typically have clear findings with
    high radiologist agreement, while "hard" samples have subtle or ambiguous
    findings.
    """

    def __init__(
        self,
        difficulty_scores: np.ndarray,
        total_epochs: int = 100,
        initial_fraction: float = 0.3,
        pacing: str = "linear",  # linear, sqrt, log, step
    ) -> None:
        super().__init__(difficulty_scores, total_epochs)
        self.initial_fraction = initial_fraction
        self.pacing = pacing

    def _compute_fraction(self, epoch: int) -> float:
        """Compute the fraction of data to include at this epoch."""
        p = epoch / max(self.state.total_epochs, 1)

        if self.pacing == "linear":
            frac = self.initial_fraction + (1.0 - self.initial_fraction) * p
        elif self.pacing == "sqrt":
            frac = self.initial_fraction + (1.0 - self.initial_fraction) * math.sqrt(p)
        elif self.pacing == "log":
            frac = self.initial_fraction + (1.0 - self.initial_fraction) * math.log1p(p) / math.log(2)
        elif self.pacing == "step":
            if p < 0.33:
                frac = self.initial_fraction
            elif p < 0.66:
                frac = 0.5 + self.initial_fraction * 0.5
            else:
                frac = 1.0
        else:
            frac = min(1.0, self.initial_fraction + p)

        return min(frac, 1.0)

    def get_sample_indices(self, epoch: int) -> np.ndarray:
        frac = self._compute_fraction(epoch)
        n_include = max(1, int(self.n_samples * frac))
        # Take the easiest n_include samples
        return self.sorted_indices[:n_include]

    def get_sample_weights(self, epoch: int) -> np.ndarray:
        """Uniform weights within the active set."""
        weights = np.zeros(self.n_samples)
        indices = self.get_sample_indices(epoch)
        weights[indices] = 1.0
        return weights


class SelfPacedCurriculum(CurriculumScheduler):
    """Self-paced learning with dynamic difficulty thresholds.

    Instead of a fixed schedule, the difficulty threshold adapts based on the
    model's current competence (measured by training loss). Samples with loss
    below a dynamic threshold are included; the threshold grows as the model
    improves.

    This is particularly useful in medical imaging where the difficulty
    distribution is unknown a priori and varies significantly across pathologies.

    Reference: Kumar et al., "Self-Paced Learning with Diversity", NeurIPS 2010
    """

    def __init__(
        self,
        difficulty_scores: np.ndarray,
        total_epochs: int = 100,
        initial_threshold: float = 0.3,
        growth_rate: float = 0.05,
        min_samples_fraction: float = 0.2,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(difficulty_scores, total_epochs)
        self.threshold = initial_threshold
        self.growth_rate = growth_rate
        self.min_samples_fraction = min_samples_fraction
        self.momentum = momentum

        # Track per-sample losses for dynamic thresholding
        self.sample_losses = np.zeros(self.n_samples)
        self._initialized = False

    def update_losses(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """Update tracked losses after a training epoch."""
        for idx, loss in zip(indices, losses):
            if self._initialized:
                self.sample_losses[idx] = (
                    self.momentum * self.sample_losses[idx]
                    + (1 - self.momentum) * loss
                )
            else:
                self.sample_losses[idx] = loss
        self._initialized = True

    def _update_threshold(self, epoch: int) -> None:
        """Dynamically adjust difficulty threshold based on model competence."""
        if not self._initialized:
            return

        mean_loss = self.sample_losses[self.sample_losses > 0].mean()
        # Threshold grows as the model gets better (lower mean loss)
        self.threshold = min(
            1.0,
            self.threshold + self.growth_rate * (1.0 - mean_loss),
        )
        self.state.current_difficulty_threshold = self.threshold

    def get_sample_indices(self, epoch: int) -> np.ndarray:
        self._update_threshold(epoch)

        # Include samples whose difficulty is below the current threshold
        mask = self.difficulty_scores <= self.threshold

        # Ensure minimum number of samples
        min_samples = max(1, int(self.n_samples * self.min_samples_fraction))
        if mask.sum() < min_samples:
            # Fall back to easiest min_samples
            indices = self.sorted_indices[:min_samples]
        else:
            indices = np.where(mask)[0]

        return indices

    def get_sample_weights(self, epoch: int) -> np.ndarray:
        """Weight samples inversely by difficulty within the active set."""
        indices = self.get_sample_indices(epoch)
        weights = np.zeros(self.n_samples)
        for idx in indices:
            # Easier samples get slightly higher weight early on, equal later
            ease = 1.0 - self.difficulty_scores[idx]
            curriculum_weight = ease * (1.0 - self.state.progress) + self.state.progress
            weights[idx] = curriculum_weight
        return weights


class AntiCurriculumScheduler(CurriculumScheduler):
    """Anti-curriculum: start with hard samples for robustness.

    Counterintuitively, starting with harder examples can improve robustness
    to distribution shift, which is important when deploying across sites.
    After the initial "hard" phase, gradually includes easier samples.

    This is motivated by the observation that models trained on hard examples
    early tend to learn more robust features, at the cost of slower convergence.
    """

    def __init__(
        self,
        difficulty_scores: np.ndarray,
        total_epochs: int = 100,
        hard_fraction: float = 0.3,
        transition_epoch: float = 0.4,
    ) -> None:
        super().__init__(difficulty_scores, total_epochs)
        self.hard_fraction = hard_fraction
        self.transition_epoch = transition_epoch

    def get_sample_indices(self, epoch: int) -> np.ndarray:
        p = epoch / max(self.state.total_epochs, 1)

        if p < self.transition_epoch:
            # Hard phase: only hardest samples
            n_hard = max(1, int(self.n_samples * self.hard_fraction))
            return self.sorted_indices[-n_hard:]  # hardest at the end
        else:
            # Transition: gradually add easier samples
            transition_progress = (p - self.transition_epoch) / (1.0 - self.transition_epoch)
            n_include = max(1, int(self.n_samples * (self.hard_fraction + (1.0 - self.hard_fraction) * transition_progress)))
            # Still start from hardest
            return self.sorted_indices[-n_include:]

    def get_sample_weights(self, epoch: int) -> np.ndarray:
        indices = self.get_sample_indices(epoch)
        weights = np.zeros(self.n_samples)
        weights[indices] = 1.0
        return weights


class TeacherStudentCurriculum(CurriculumScheduler):
    """Teacher-student curriculum with a mentor model guiding sample selection.

    A pretrained "teacher" model (e.g., larger model, ensemble, or model
    from prior training run) scores samples by prediction confidence.
    The student sees samples in order of teacher confidence -- starting with
    samples the teacher is most confident about.

    This is especially useful when transferring knowledge from a model trained
    on a large dataset (e.g., natural images) to a smaller medical dataset.
    """

    def __init__(
        self,
        difficulty_scores: np.ndarray,
        total_epochs: int = 100,
        teacher_confidence: Optional[np.ndarray] = None,
        confidence_weight: float = 0.5,
        initial_fraction: float = 0.4,
    ) -> None:
        super().__init__(difficulty_scores, total_epochs)
        self.confidence_weight = confidence_weight
        self.initial_fraction = initial_fraction

        if teacher_confidence is not None:
            self.teacher_confidence = teacher_confidence
        else:
            # If no teacher scores, fall back to inverse difficulty
            self.teacher_confidence = 1.0 - difficulty_scores

        # Combined score: weighted average of difficulty and teacher confidence
        self.combined_scores = (
            (1.0 - confidence_weight) * (1.0 - difficulty_scores)
            + confidence_weight * self.teacher_confidence
        )
        self.combined_sorted = np.argsort(-self.combined_scores)  # descending

    def get_sample_indices(self, epoch: int) -> np.ndarray:
        p = epoch / max(self.state.total_epochs, 1)
        frac = self.initial_fraction + (1.0 - self.initial_fraction) * p
        n_include = max(1, int(self.n_samples * min(frac, 1.0)))
        return self.combined_sorted[:n_include]

    def get_sample_weights(self, epoch: int) -> np.ndarray:
        indices = self.get_sample_indices(epoch)
        weights = np.zeros(self.n_samples)
        for idx in indices:
            weights[idx] = self.combined_scores[idx]
        return weights

    @staticmethod
    @torch.no_grad()
    def compute_teacher_confidence(
        teacher: nn.Module,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Compute per-sample confidence scores from a teacher model.

        Confidence is measured as the max softmax probability, which
        correlates with prediction certainty.
        """
        teacher.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        confidences = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            logits = teacher(images)
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidences.append(max_probs.cpu().numpy())

        return np.concatenate(confidences)


# ---------------------------------------------------------------------------
# Curriculum Builder
# ---------------------------------------------------------------------------

def build_curriculum(
    strategy: str,
    difficulty_scores: np.ndarray,
    total_epochs: int = 100,
    **kwargs: Any,
) -> CurriculumScheduler:
    """Factory function to build a curriculum scheduler.

    Args:
        strategy: One of 'difficulty', 'self_paced', 'anti_curriculum', 'teacher_student'
        difficulty_scores: Per-sample difficulty scores in [0, 1]
        total_epochs: Total training epochs
        **kwargs: Strategy-specific parameters

    Returns:
        Configured CurriculumScheduler
    """
    builders = {
        "difficulty": DifficultyBasedCurriculum,
        "self_paced": SelfPacedCurriculum,
        "anti_curriculum": AntiCurriculumScheduler,
        "teacher_student": TeacherStudentCurriculum,
    }

    if strategy not in builders:
        raise ValueError(f"Unknown curriculum strategy: {strategy}. Available: {list(builders.keys())}")

    scheduler = builders[strategy](
        difficulty_scores=difficulty_scores,
        total_epochs=total_epochs,
        **kwargs,
    )
    logger.info(
        f"Built {strategy} curriculum for {len(difficulty_scores)} samples, "
        f"{total_epochs} epochs"
    )
    return scheduler

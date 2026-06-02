"""
Unit tests for the Medical Image Foundation Model pipeline.

Tests cover core functionality without requiring GPU or real medical data:
- Model construction and forward pass
- Channel adaptation strategies
- Domain adaptation losses
- Curriculum scheduling logic
- Difficulty scoring
- Few-shot episode sampling
- Domain gap metrics
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Foundation Model Tests
# ---------------------------------------------------------------------------

class TestChannelAdapter:
    """Test single-channel adaptation for medical images."""

    def test_replicate_strategy(self):
        from src.foundation.model_zoo import ChannelAdapter

        original = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        adapter = ChannelAdapter(original, strategy="replicate")

        x = torch.randn(2, 1, 224, 224)
        output = adapter(x)
        assert output.shape == (2, 64, 112, 112)

    def test_average_strategy(self):
        from src.foundation.model_zoo import ChannelAdapter

        original = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        adapter = ChannelAdapter(original, strategy="average")

        x = torch.randn(2, 1, 224, 224)
        output = adapter(x)
        assert output.shape == (2, 64, 112, 112)

    def test_learned_strategy(self):
        from src.foundation.model_zoo import ChannelAdapter

        original = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        adapter = ChannelAdapter(original, strategy="learned")

        x = torch.randn(2, 1, 224, 224)
        output = adapter(x)
        assert output.shape == (2, 64, 112, 112)


class TestResolutionAdapter:
    """Test resolution adaptation for high-res medical images."""

    def test_interpolation(self):
        from src.foundation.model_zoo import ResolutionAdapter

        adapter = ResolutionAdapter(target_resolution=512, pretrained_resolution=224)
        x = torch.randn(2, 1, 512, 512)
        output = adapter(x)
        assert output.shape == (2, 1, 224, 224)

    def test_multi_scale(self):
        from src.foundation.model_zoo import ResolutionAdapter

        adapter = ResolutionAdapter(
            target_resolution=512, pretrained_resolution=224, mode="multi_scale"
        )
        x = torch.randn(2, 1, 512, 512)
        output = adapter(x)
        assert output.shape == (2, 1, 224, 224)


class TestFeaturePyramid:
    """Test Feature Pyramid Network."""

    def test_fpn_forward(self):
        from src.foundation.model_zoo import FeaturePyramidExtractor

        fpn = FeaturePyramidExtractor(feature_dims=[64, 128, 256, 512], output_dim=128)
        features = [
            torch.randn(2, 64, 56, 56),
            torch.randn(2, 128, 28, 28),
            torch.randn(2, 256, 14, 14),
            torch.randn(2, 512, 7, 7),
        ]
        outputs = fpn(features)
        assert len(outputs) == 4
        for out in outputs:
            assert out.shape[1] == 128


# ---------------------------------------------------------------------------
# Domain Adaptation Tests
# ---------------------------------------------------------------------------

class TestGradientReversal:
    """Test gradient reversal layer."""

    def test_forward_unchanged(self):
        from src.adaptation.domain_adapter import gradient_reversal

        x = torch.randn(4, 10, requires_grad=True)
        y = gradient_reversal(x, lambda_val=1.0)
        assert torch.allclose(x, y)

    def test_gradient_reversed(self):
        from src.adaptation.domain_adapter import gradient_reversal

        x = torch.randn(4, 10, requires_grad=True)
        y = gradient_reversal(x, lambda_val=1.0)
        loss = y.sum()
        loss.backward()
        # Gradient should be -1 (reversed)
        assert torch.allclose(x.grad, -torch.ones_like(x))


class TestMMDLoss:
    """Test Maximum Mean Discrepancy loss."""

    def test_same_distribution(self):
        from src.adaptation.domain_adapter import MMDLoss

        mmd = MMDLoss()
        x = torch.randn(50, 10)
        loss = mmd(x, x)
        # MMD of identical distributions should be approximately 0
        assert loss.item() < 0.1

    def test_different_distributions(self):
        from src.adaptation.domain_adapter import MMDLoss

        mmd = MMDLoss()
        x = torch.randn(100, 10)
        y = torch.randn(100, 10) + 5.0  # shifted
        loss = mmd(x, y)
        assert loss.item() > 0.0


class TestAdaptationState:
    """Test adaptation state and lambda scheduling."""

    def test_linear_schedule(self):
        from src.adaptation.domain_adapter import AdaptationState

        state = AdaptationState(total_epochs=100, lambda_schedule="linear", max_lambda=1.0)
        state.current_epoch = 0
        assert state.current_lambda == 0.0
        state.current_epoch = 50
        assert abs(state.current_lambda - 0.5) < 1e-5
        state.current_epoch = 100
        assert abs(state.current_lambda - 1.0) < 1e-5

    def test_exp_schedule(self):
        from src.adaptation.domain_adapter import AdaptationState

        state = AdaptationState(total_epochs=100, lambda_schedule="exp", max_lambda=1.0)
        state.current_epoch = 0
        assert state.current_lambda == 0.0
        state.current_epoch = 100
        assert state.current_lambda > 0.9


# ---------------------------------------------------------------------------
# Style Transfer Tests
# ---------------------------------------------------------------------------

class TestFourierDomainAdaptation:
    """Test FDA style transfer."""

    def test_output_shape(self):
        from src.adaptation.style_transfer import FourierDomainAdaptation

        fda = FourierDomainAdaptation(beta=0.05)
        source = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)
        output = fda(source, target)
        assert output.shape == source.shape

    def test_output_range(self):
        from src.adaptation.style_transfer import FourierDomainAdaptation

        fda = FourierDomainAdaptation(beta=0.05)
        source = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)
        output = fda(source, target)
        assert output.min() >= 0.0
        assert output.max() <= 1.0


# ---------------------------------------------------------------------------
# Curriculum Learning Tests
# ---------------------------------------------------------------------------

class TestCurriculumSchedulers:
    """Test curriculum learning schedulers."""

    def setup_method(self):
        np.random.seed(42)
        self.n_samples = 100
        self.difficulty_scores = np.random.rand(self.n_samples)

    def test_difficulty_based(self):
        from src.curriculum.curriculum_scheduler import DifficultyBasedCurriculum

        scheduler = DifficultyBasedCurriculum(
            self.difficulty_scores, total_epochs=100, initial_fraction=0.3
        )
        indices_early = scheduler.get_sample_indices(0)
        indices_late = scheduler.get_sample_indices(99)

        assert len(indices_early) < len(indices_late)
        assert len(indices_early) >= int(self.n_samples * 0.3)

    def test_self_paced(self):
        from src.curriculum.curriculum_scheduler import SelfPacedCurriculum

        scheduler = SelfPacedCurriculum(
            self.difficulty_scores, total_epochs=100, min_samples_fraction=0.2
        )
        indices = scheduler.get_sample_indices(0)
        assert len(indices) >= int(self.n_samples * 0.2)

    def test_anti_curriculum(self):
        from src.curriculum.curriculum_scheduler import AntiCurriculumScheduler

        scheduler = AntiCurriculumScheduler(
            self.difficulty_scores, total_epochs=100, hard_fraction=0.3
        )
        indices_early = scheduler.get_sample_indices(0)
        indices_late = scheduler.get_sample_indices(99)

        # Early indices should be harder on average
        early_mean_diff = self.difficulty_scores[indices_early].mean()
        late_mean_diff = self.difficulty_scores[indices_late].mean()
        assert early_mean_diff >= late_mean_diff - 0.1  # Allow tolerance


class TestDifficultyScorer:
    """Test difficulty scoring methods."""

    def test_data_complexity_scorer(self):
        from src.curriculum.difficulty_scorer import DataComplexityScorer

        images = np.random.rand(20, 64, 64).astype(np.float32)
        labels = np.random.randint(0, 2, size=20)
        scorer = DataComplexityScorer(images, labels)
        report = scorer.score()

        assert len(report.scores) == 20
        assert report.scores.min() >= 0.0
        assert report.scores.max() <= 1.0

    def test_radiologist_agreement(self):
        from src.curriculum.difficulty_scorer import RadiologistAgreementScorer

        # 3 readers, 10 samples
        annotations = np.array([
            [0, 0, 0],  # full agreement -> easy
            [1, 1, 1],  # full agreement -> easy
            [0, 1, 0],  # partial agreement -> medium
            [0, 1, 1],  # partial agreement -> medium
            [0, 0, 1],  # partial agreement -> medium
            [1, 0, 1],  # partial agreement -> medium
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
        ])
        scorer = RadiologistAgreementScorer(annotations)
        report = scorer.score()

        assert len(report.scores) == 10
        # Full agreement samples should have lower difficulty
        assert report.scores[0] < report.scores[2]


# ---------------------------------------------------------------------------
# Few-Shot Learning Tests
# ---------------------------------------------------------------------------

class TestEpisodeSampler:
    """Test few-shot episode sampling."""

    def test_episode_shapes(self):
        from src.transfer.few_shot import EpisodeConfig, EpisodeSampler

        config = EpisodeConfig(n_way=3, n_support=2, n_query=4, n_episodes=10)
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20)
        sampler = EpisodeSampler(labels, config)

        support_idx, support_labels, query_idx, query_labels = sampler.sample_episode()

        assert len(support_idx) == 3 * 2  # n_way * n_support
        assert len(query_idx) == 3 * 4    # n_way * n_query
        assert set(support_labels) == set(query_labels)
        assert len(set(support_labels)) == 3  # n_way classes


class TestSupportSetAugmentor:
    """Test support set augmentation."""

    def test_augmentation_factor(self):
        from src.transfer.few_shot import SupportSetAugmentor

        augmentor = SupportSetAugmentor(
            augmentation_factor=5,
            include_elastic=False,  # Skip for speed
        )
        images = torch.rand(3, 1, 32, 32)
        labels = torch.tensor([0, 1, 2])

        aug_images, aug_labels = augmentor.augment_support_set(images, labels)
        assert aug_images.shape[0] == 15  # 3 * 5
        assert aug_labels.shape[0] == 15


# ---------------------------------------------------------------------------
# Domain Gap Tests
# ---------------------------------------------------------------------------

class TestFIDCalculator:
    """Test FID computation."""

    def test_identical_distributions(self):
        from src.evaluation.domain_gap import FIDCalculator

        calc = FIDCalculator()
        features = np.random.randn(100, 32)
        fid = calc(features, features)
        assert fid < 1.0

    def test_different_distributions(self):
        from src.evaluation.domain_gap import FIDCalculator

        calc = FIDCalculator()
        f1 = np.random.randn(200, 32)
        f2 = np.random.randn(200, 32) + 3.0
        fid = calc(f1, f2)
        assert fid > 1.0


class TestCKA:
    """Test Centered Kernel Alignment."""

    def test_identical_representations(self):
        from src.evaluation.transfer_analysis import CKACalculator

        X = np.random.randn(50, 16)
        cka = CKACalculator.linear_cka(X, X)
        assert abs(cka - 1.0) < 1e-5

    def test_orthogonal_representations(self):
        from src.evaluation.transfer_analysis import CKACalculator

        np.random.seed(42)
        X = np.random.randn(100, 16)
        Y = np.random.randn(100, 16)
        cka = CKACalculator.linear_cka(X, Y)
        # Independent random matrices should have low CKA
        assert cka < 0.3


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestHistogramMatcher:
    """Test histogram matching."""

    def test_numpy_matching(self):
        from src.adaptation.style_transfer import HistogramMatcher

        matcher = HistogramMatcher(n_bins=64)
        reference = np.random.rand(10, 64, 64).astype(np.float32)
        matcher.fit_reference(reference)

        source = np.random.rand(64, 64).astype(np.float32)
        matched = matcher.match(source)

        assert matched.shape == source.shape
        assert matched.min() >= 0.0
        assert matched.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

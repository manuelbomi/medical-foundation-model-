"""
Domain Gap Measurement Between Source and Target Datasets.

Quantifies the distribution shift between pretrained (source) and medical
(target) image domains using multiple complementary metrics. These
measurements guide adaptation strategy selection and provide an objective
measure of how much domain shift exists.

Metrics:
- Frechet Inception Distance (FID): Gaussian fit to feature distributions
- Maximum Mean Discrepancy (MMD): Kernel-based distribution distance
- A-distance (proxy): Accuracy of a domain classifier as a distance measure
- Feature distribution visualization (t-SNE, PCA)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class DomainGapReport:
    """Report containing all domain gap metrics."""
    fid: Optional[float] = None
    mmd: Optional[float] = None
    a_distance: Optional[float] = None
    source_stats: Optional[Dict[str, float]] = None
    target_stats: Optional[Dict[str, float]] = None
    feature_dim: int = 0
    n_source: int = 0
    n_target: int = 0

    def summary(self) -> str:
        lines = [
            "Domain Gap Report",
            f"  Source samples: {self.n_source}",
            f"  Target samples: {self.n_target}",
            f"  Feature dim: {self.feature_dim}",
        ]
        if self.fid is not None:
            lines.append(f"  FID: {self.fid:.4f}")
        if self.mmd is not None:
            lines.append(f"  MMD: {self.mmd:.6f}")
        if self.a_distance is not None:
            lines.append(f"  A-distance: {self.a_distance:.4f}")
        return "\n".join(lines)


class FeatureExtractor:
    """Extract feature representations from a model for domain gap analysis."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.layer_name = layer_name
        self._features: List[torch.Tensor] = []
        self._hook_handle = None

        if layer_name is not None:
            self._register_hook(layer_name)

    def _register_hook(self, layer_name: str) -> None:
        """Register a forward hook to capture intermediate features."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                self._hook_handle = module.register_forward_hook(self._hook_fn)
                logger.info(f"Registered feature hook on layer: {layer_name}")
                return
        raise ValueError(f"Layer '{layer_name}' not found in model")

    def _hook_fn(
        self, module: nn.Module, input: Any, output: torch.Tensor
    ) -> None:
        self._features.append(output.detach())

    @torch.no_grad()
    def extract(
        self, dataloader: DataLoader, max_batches: int = 200
    ) -> np.ndarray:
        """Extract features from a dataset.

        Returns:
            Feature array (n_samples, feature_dim)
        """
        self.model.eval()
        self._features = []
        all_features = []

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(self.device)

            output = self.model(images)

            if self._hook_handle is not None:
                # Use hooked features
                feat = self._features[-1]
            else:
                # Use model output
                if isinstance(output, tuple):
                    feat = output[0]
                else:
                    feat = output

            # Pool spatial dimensions
            if feat.dim() == 4:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)

            all_features.append(feat.cpu().numpy())

        self._features = []
        return np.concatenate(all_features, axis=0)

    def cleanup(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ---------------------------------------------------------------------------
# FID (Frechet Inception Distance)
# ---------------------------------------------------------------------------

class FIDCalculator:
    """Compute Frechet Inception Distance between two feature distributions.

    FID models each distribution as a multivariate Gaussian and computes
    the Frechet distance (Wasserstein-2) between them. Lower FID indicates
    more similar distributions.

    Reference: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
    Converge to a Local Nash Equilibrium", NeurIPS 2017
    """

    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features."""
        mu = features.mean(axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    @staticmethod
    def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root using eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        # Clip negative eigenvalues (numerical issues)
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return (eigenvectors * sqrt_eigenvalues) @ eigenvectors.T

    @staticmethod
    def compute_fid(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """Compute FID between two Gaussian distributions.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
        """
        diff = mu1 - mu2
        mean_diff_sq = np.sum(diff ** 2)

        # Product of covariance matrices
        cov_product = sigma1 @ sigma2
        sqrt_cov = FIDCalculator._matrix_sqrt(cov_product)

        trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_cov)

        fid = mean_diff_sq + trace_term
        return float(max(fid, 0.0))  # Clip numerical negatives

    def __call__(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> float:
        mu1, sigma1 = self.compute_statistics(source_features)
        mu2, sigma2 = self.compute_statistics(target_features)
        return self.compute_fid(mu1, sigma1, mu2, sigma2)


# ---------------------------------------------------------------------------
# MMD (for domain gap measurement)
# ---------------------------------------------------------------------------

class MMDCalculator:
    """Compute Maximum Mean Discrepancy between distributions.

    Uses a mixture of Gaussian kernels for robustness across scales.
    """

    def __init__(self, kernel_bandwidths: Optional[List[float]] = None) -> None:
        if kernel_bandwidths is None:
            kernel_bandwidths = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.kernel_bandwidths = kernel_bandwidths

    def __call__(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> float:
        """Compute MMD^2 between source and target."""
        # Subsample for computational efficiency
        max_samples = 2000
        if len(source_features) > max_samples:
            idx = np.random.choice(len(source_features), max_samples, replace=False)
            source_features = source_features[idx]
        if len(target_features) > max_samples:
            idx = np.random.choice(len(target_features), max_samples, replace=False)
            target_features = target_features[idx]

        src = torch.from_numpy(source_features).float()
        tgt = torch.from_numpy(target_features).float()

        mmd = 0.0
        for bw in self.kernel_bandwidths:
            k_ss = self._gaussian_kernel(src, src, bw).mean().item()
            k_tt = self._gaussian_kernel(tgt, tgt, bw).mean().item()
            k_st = self._gaussian_kernel(src, tgt, bw).mean().item()
            mmd += k_ss + k_tt - 2.0 * k_st

        return mmd / len(self.kernel_bandwidths)

    @staticmethod
    def _gaussian_kernel(
        x: torch.Tensor, y: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        y_sq = (y ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_sq + y_sq.T - 2.0 * (x @ y.T)
        return torch.exp(-dist_sq / (2.0 * bandwidth))


# ---------------------------------------------------------------------------
# A-distance (Proxy)
# ---------------------------------------------------------------------------

class ADistanceCalculator:
    """Compute proxy A-distance using a linear domain classifier.

    A-distance = 2 * (1 - 2 * error), where error is the generalization
    error of a linear classifier trained to distinguish source from target.

    High A-distance indicates large domain gap; low A-distance means the
    domains are hard to distinguish (well-aligned features).

    Reference: Ben-David et al., "A Theory of Learning from Different Domains", MLJ 2010
    """

    def __init__(self, n_epochs: int = 50, hidden_dim: int = 256) -> None:
        self.n_epochs = n_epochs
        self.hidden_dim = hidden_dim

    def __call__(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> float:
        """Compute proxy A-distance."""
        n_src = len(source_features)
        n_tgt = len(target_features)

        # Create binary classification dataset
        X = np.concatenate([source_features, target_features], axis=0)
        y = np.concatenate([np.zeros(n_src), np.ones(n_tgt)])

        # Shuffle
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        # Split train/val (80/20)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).float()
        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val).float()

        # Simple linear classifier
        feature_dim = X.shape[1]
        classifier = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        # Train
        classifier.train()
        batch_size = min(256, len(X_train))
        for epoch in range(self.n_epochs):
            perm_t = torch.randperm(len(X_train_t))
            for start in range(0, len(X_train_t), batch_size):
                end = min(start + batch_size, len(X_train_t))
                idx = perm_t[start:end]
                logits = classifier(X_train_t[idx]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_train_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val_t).squeeze(-1)
            val_preds = (val_logits > 0).float()
            error = (val_preds != y_val_t).float().mean().item()

        a_distance = 2.0 * (1.0 - 2.0 * error)
        return float(max(a_distance, 0.0))


# ---------------------------------------------------------------------------
# Feature Visualization
# ---------------------------------------------------------------------------

class FeatureVisualizer:
    """Compute low-dimensional projections for visualization.

    Provides t-SNE and PCA projections of source/target features for
    visual inspection of domain gap.
    """

    @staticmethod
    def pca_projection(
        features: np.ndarray, n_components: int = 2
    ) -> np.ndarray:
        """Simple PCA projection (no sklearn dependency)."""
        centered = features - features.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Take top n_components
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        components = eigenvectors[:, idx]

        return centered @ components

    @staticmethod
    def compute_feature_statistics(features: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics of feature distributions."""
        return {
            "mean_norm": float(np.linalg.norm(features, axis=1).mean()),
            "std_norm": float(np.linalg.norm(features, axis=1).std()),
            "mean_activation": float(features.mean()),
            "sparsity": float((np.abs(features) < 0.01).mean()),
            "effective_rank": float(
                np.exp(-np.sum(
                    (s := np.linalg.svd(features - features.mean(0), compute_uv=False) / len(features))
                    / s.sum() * np.log(s / s.sum() + 1e-10)
                ))
            ) if len(features) > 1 else 0.0,
        }


# ---------------------------------------------------------------------------
# Domain Gap Analyzer (Orchestrator)
# ---------------------------------------------------------------------------

class DomainGapAnalyzer:
    """High-level analyzer that computes all domain gap metrics.

    Coordinates feature extraction, metric computation, and visualization
    into a single analysis pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        feature_layer: Optional[str] = None,
    ) -> None:
        self.extractor = FeatureExtractor(model, device, feature_layer)
        self.fid_calc = FIDCalculator()
        self.mmd_calc = MMDCalculator()
        self.a_distance_calc = ADistanceCalculator()
        self.visualizer = FeatureVisualizer()

    def analyze(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        max_batches: int = 200,
        compute_fid: bool = True,
        compute_mmd: bool = True,
        compute_a_distance: bool = True,
    ) -> DomainGapReport:
        """Run full domain gap analysis.

        Args:
            source_loader: DataLoader for source (e.g., ImageNet) domain
            target_loader: DataLoader for target (e.g., mammography) domain
            max_batches: Max batches to process per domain
            compute_fid: Whether to compute FID
            compute_mmd: Whether to compute MMD
            compute_a_distance: Whether to compute A-distance

        Returns:
            DomainGapReport with all requested metrics
        """
        logger.info("Extracting source features...")
        source_features = self.extractor.extract(source_loader, max_batches)
        logger.info(f"Source features shape: {source_features.shape}")

        logger.info("Extracting target features...")
        target_features = self.extractor.extract(target_loader, max_batches)
        logger.info(f"Target features shape: {target_features.shape}")

        report = DomainGapReport(
            feature_dim=source_features.shape[1],
            n_source=len(source_features),
            n_target=len(target_features),
        )

        if compute_fid:
            logger.info("Computing FID...")
            report.fid = self.fid_calc(source_features, target_features)
            logger.info(f"FID: {report.fid:.4f}")

        if compute_mmd:
            logger.info("Computing MMD...")
            report.mmd = self.mmd_calc(source_features, target_features)
            logger.info(f"MMD: {report.mmd:.6f}")

        if compute_a_distance:
            logger.info("Computing A-distance...")
            report.a_distance = self.a_distance_calc(source_features, target_features)
            logger.info(f"A-distance: {report.a_distance:.4f}")

        # Feature statistics
        report.source_stats = self.visualizer.compute_feature_statistics(source_features)
        report.target_stats = self.visualizer.compute_feature_statistics(target_features)

        self.extractor.cleanup()
        return report

    def get_projections(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        max_batches: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get PCA projections for visualization.

        Returns:
            source_2d, target_2d: 2D projections
            labels: 0 for source, 1 for target
        """
        source_features = self.extractor.extract(source_loader, max_batches)
        target_features = self.extractor.extract(target_loader, max_batches)

        combined = np.concatenate([source_features, target_features], axis=0)
        projected = self.visualizer.pca_projection(combined, n_components=2)

        n_src = len(source_features)
        labels = np.concatenate([np.zeros(n_src), np.ones(len(target_features))])

        self.extractor.cleanup()
        return projected[:n_src], projected[n_src:], labels

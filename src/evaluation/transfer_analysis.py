"""
Transfer Learning Effectiveness Analysis.

Provides tools for understanding how well features transfer from pretrained
models to medical imaging tasks, including:
- CKA (Centered Kernel Alignment) for layer-wise similarity
- Feature transferability scoring
- Fine-tuning vs linear probe comparison
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TransferAnalysisReport:
    """Report on transfer learning effectiveness."""
    cka_matrix: Optional[np.ndarray] = None
    layer_names: List[str] = field(default_factory=list)
    transferability_scores: Optional[Dict[str, float]] = None
    linear_probe_acc: Optional[float] = None
    fine_tune_acc: Optional[float] = None
    transfer_gain: Optional[float] = None

    def summary(self) -> str:
        lines = ["Transfer Analysis Report"]
        if self.transferability_scores:
            lines.append("  Transferability scores:")
            for name, score in self.transferability_scores.items():
                lines.append(f"    {name}: {score:.4f}")
        if self.linear_probe_acc is not None:
            lines.append(f"  Linear probe accuracy: {self.linear_probe_acc:.4f}")
        if self.fine_tune_acc is not None:
            lines.append(f"  Fine-tune accuracy: {self.fine_tune_acc:.4f}")
        if self.transfer_gain is not None:
            lines.append(f"  Transfer gain: {self.transfer_gain:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

class CKACalculator:
    """Centered Kernel Alignment for comparing neural network representations.

    CKA measures the similarity between two sets of representations,
    invariant to orthogonal transformations and isotropic scaling.
    This makes it ideal for comparing features across:
    - Different layers of the same model
    - Same layer before/after fine-tuning
    - Corresponding layers across different architectures

    A CKA value of 1.0 means identical representations (up to rotation/scale);
    0.0 means completely unrelated.

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019
    """

    @staticmethod
    def _center_gram(gram: np.ndarray) -> np.ndarray:
        """Center a Gram matrix (double centering)."""
        n = gram.shape[0]
        row_mean = gram.mean(axis=1, keepdims=True)
        col_mean = gram.mean(axis=0, keepdims=True)
        total_mean = gram.mean()
        return gram - row_mean - col_mean + total_mean

    @staticmethod
    def _hsic(K: np.ndarray, L: np.ndarray) -> float:
        """Compute Hilbert-Schmidt Independence Criterion."""
        K_c = CKACalculator._center_gram(K)
        L_c = CKACalculator._center_gram(L)
        n = K.shape[0]
        return float(np.trace(K_c @ L_c) / ((n - 1) ** 2))

    @staticmethod
    def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA between two representation matrices.

        Args:
            X: (n_samples, feature_dim_1) representations from model/layer 1
            Y: (n_samples, feature_dim_2) representations from model/layer 2

        Returns:
            CKA similarity in [0, 1]
        """
        # Gram matrices using linear kernel
        K = X @ X.T
        L = Y @ Y.T

        hsic_kl = CKACalculator._hsic(K, L)
        hsic_kk = CKACalculator._hsic(K, K)
        hsic_ll = CKACalculator._hsic(L, L)

        denominator = np.sqrt(hsic_kk * hsic_ll)
        if denominator < 1e-10:
            return 0.0
        return float(hsic_kl / denominator)

    @staticmethod
    def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
        """Compute CKA with RBF (Gaussian) kernel."""
        def rbf_kernel(Z: np.ndarray, sigma: float) -> np.ndarray:
            sq_dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
            return np.exp(-sq_dists / (2 * sigma ** 2))

        K = rbf_kernel(X, sigma)
        L = rbf_kernel(Y, sigma)

        hsic_kl = CKACalculator._hsic(K, L)
        hsic_kk = CKACalculator._hsic(K, K)
        hsic_ll = CKACalculator._hsic(L, L)

        denominator = np.sqrt(hsic_kk * hsic_ll)
        if denominator < 1e-10:
            return 0.0
        return float(hsic_kl / denominator)


class LayerWiseCKA:
    """Compute CKA between all layer pairs of two models.

    This produces a matrix where entry (i, j) is the CKA between
    layer i of model 1 and layer j of model 2. The diagonal of this
    matrix (when comparing pretrained vs fine-tuned) shows which layers
    changed the most during adaptation.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cka = CKACalculator()

    def _extract_layer_features(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
        max_batches: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Extract features from specified layers."""
        model.eval()
        features: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
                feat = output.detach()
                if feat.dim() == 4:
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                elif feat.dim() == 3:
                    feat = feat.mean(dim=1)
                features[name].append(feat.cpu())
            return hook_fn

        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device)
                model(images)

        for h in hooks:
            h.remove()

        return {
            name: torch.cat(feat_list, dim=0).numpy()
            for name, feat_list in features.items()
            if feat_list
        }

    def compute_cka_matrix(
        self,
        model1: nn.Module,
        model2: nn.Module,
        dataloader: DataLoader,
        layer_names1: List[str],
        layer_names2: Optional[List[str]] = None,
        max_batches: int = 50,
        kernel: str = "linear",
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Compute full CKA matrix between two models.

        Returns:
            cka_matrix: (n_layers_1, n_layers_2) matrix of CKA values
            layer_names1: Names of model 1 layers
            layer_names2: Names of model 2 layers
        """
        if layer_names2 is None:
            layer_names2 = layer_names1

        logger.info(f"Extracting features from model 1 ({len(layer_names1)} layers)...")
        features1 = self._extract_layer_features(model1, dataloader, layer_names1, max_batches)

        logger.info(f"Extracting features from model 2 ({len(layer_names2)} layers)...")
        features2 = self._extract_layer_features(model2, dataloader, layer_names2, max_batches)

        active_names1 = [n for n in layer_names1 if n in features1]
        active_names2 = [n for n in layer_names2 if n in features2]

        cka_matrix = np.zeros((len(active_names1), len(active_names2)))

        cka_fn = self.cka.linear_cka if kernel == "linear" else self.cka.rbf_cka

        for i, name1 in enumerate(active_names1):
            for j, name2 in enumerate(active_names2):
                n = min(len(features1[name1]), len(features2[name2]))
                cka_matrix[i, j] = cka_fn(features1[name1][:n], features2[name2][:n])

        return cka_matrix, active_names1, active_names2


# ---------------------------------------------------------------------------
# Feature Transferability Scoring
# ---------------------------------------------------------------------------

class TransferabilityScorer:
    """Score how transferable features are from source to target task.

    Uses multiple heuristics to estimate transfer potential without
    actually fine-tuning:
    - H-score: Measures feature discriminability and redundancy
    - Log Expected Empirical Prediction (LEEP)
    - Negative Conditional Entropy (NCE)
    """

    @staticmethod
    def h_score(features: np.ndarray, labels: np.ndarray) -> float:
        """Compute H-score transferability metric.

        H-score = tr(cov_between) / tr(cov_total) -- the ratio of between-class
        to total variance. Higher values indicate more discriminative features.

        Reference: Bao et al., "An Information-Theoretic Approach to Transferability
        in Task Transfer Learning", ICIP 2019
        """
        n_classes = len(np.unique(labels))
        global_mean = features.mean(axis=0)
        total_cov = np.cov(features, rowvar=False)

        between_cov = np.zeros_like(total_cov)
        for cls in np.unique(labels):
            mask = labels == cls
            n_cls = mask.sum()
            cls_mean = features[mask].mean(axis=0)
            diff = (cls_mean - global_mean).reshape(-1, 1)
            between_cov += n_cls * (diff @ diff.T)
        between_cov /= len(features)

        # Regularize for numerical stability
        total_cov += 1e-6 * np.eye(total_cov.shape[0])

        score = np.trace(np.linalg.solve(total_cov, between_cov))
        return float(score)

    @staticmethod
    def leep_score(
        source_probs: np.ndarray, target_labels: np.ndarray
    ) -> float:
        """Log Expected Empirical Prediction (LEEP).

        Estimates transfer performance by computing the expected log-likelihood
        of target labels under the source model's predictions.

        Args:
            source_probs: (n_samples, n_source_classes) predicted probabilities
            target_labels: (n_samples,) target class labels

        Reference: Nguyen et al., "LEEP: A New Measure to Evaluate Transferability
        of Learned Representations", ICML 2020
        """
        n_samples = len(target_labels)
        n_source_classes = source_probs.shape[1]
        target_classes = np.unique(target_labels)

        # Compute empirical conditional P(target_class | source_class)
        joint = np.zeros((len(target_classes), n_source_classes))
        for i, t_cls in enumerate(target_classes):
            mask = target_labels == t_cls
            joint[i] = source_probs[mask].sum(axis=0) / n_samples

        # P(source_class) = marginal
        p_source = source_probs.mean(axis=0)

        # P(target | source) = joint / marginal
        conditional = joint / (p_source[None, :] + 1e-10)

        # LEEP = (1/n) * sum_i log(sum_z P(y_i|z) * P(z|x_i))
        leep = 0.0
        for i in range(n_samples):
            t_idx = np.where(target_classes == target_labels[i])[0][0]
            prob = (conditional[t_idx] * source_probs[i]).sum()
            leep += np.log(prob + 1e-10)

        return float(leep / n_samples)

    @staticmethod
    def nce_score(features: np.ndarray, labels: np.ndarray) -> float:
        """Negative Conditional Entropy score.

        Measures how well features predict labels. Lower conditional entropy
        (higher NCE) means more transferable features.
        """
        n_samples = len(labels)
        n_classes = len(np.unique(labels))

        # Simple KNN-based conditional entropy estimate
        # For each sample, find its nearest neighbors and check label consistency
        from sklearn.neighbors import NearestNeighbors
        k = min(10, n_samples - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn_model.fit(features)
        _, indices = nn_model.kneighbors(features)

        # Compute conditional entropy H(Y|X) using KNN labels
        cond_entropy = 0.0
        for i in range(n_samples):
            neighbor_labels = labels[indices[i, 1:]]  # exclude self
            counts = np.bincount(neighbor_labels, minlength=n_classes)
            probs = counts / k
            probs = probs[probs > 0]
            cond_entropy -= np.sum(probs * np.log(probs))

        cond_entropy /= n_samples
        return float(-cond_entropy)


# ---------------------------------------------------------------------------
# Linear Probe vs Fine-Tuning Comparison
# ---------------------------------------------------------------------------

class TransferComparison:
    """Compare linear probing vs full fine-tuning to assess feature quality.

    Linear probing (frozen backbone + trained head) measures how useful
    the pretrained features are "as-is". Fine-tuning measures the total
    achievable performance. The gap between them indicates how much
    adaptation is needed.

    Small gap: Features transfer well, minimal adaptation needed
    Large gap: Significant domain shift, full adaptation pipeline recommended
    """

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        num_classes: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device

    @torch.no_grad()
    def _extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from a dataset."""
        self.model.eval()
        features_list = []
        labels_list = []

        for batch in dataloader:
            images, labels = batch[0].to(self.device), batch[1]
            output = self.model(images)
            if isinstance(output, tuple):
                feat = output[0]
            else:
                feat = output
            if feat.dim() == 4:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

        return np.concatenate(features_list), np.concatenate(labels_list)

    def linear_probe(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> float:
        """Train and evaluate a linear probe on frozen features."""
        logger.info("Running linear probe evaluation...")

        train_features, train_labels = self._extract_features(train_loader)
        val_features, val_labels = self._extract_features(val_loader)

        # Train linear classifier
        classifier = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        X_train = torch.from_numpy(train_features).float().to(self.device)
        y_train = torch.from_numpy(train_labels).long().to(self.device)

        classifier.train()
        for epoch in range(n_epochs):
            logits = classifier(X_train)
            loss = F.cross_entropy(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        classifier.eval()
        X_val = torch.from_numpy(val_features).float().to(self.device)
        y_val = torch.from_numpy(val_labels).long().to(self.device)

        with torch.no_grad():
            val_logits = classifier(X_val)
            preds = val_logits.argmax(dim=-1)
            accuracy = (preds == y_val).float().mean().item()

        logger.info(f"Linear probe accuracy: {accuracy:.4f}")
        return accuracy

    def compare(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fine_tune_accuracy: Optional[float] = None,
    ) -> TransferAnalysisReport:
        """Run comparison and generate report."""
        probe_acc = self.linear_probe(train_loader, val_loader)

        report = TransferAnalysisReport(
            linear_probe_acc=probe_acc,
            fine_tune_acc=fine_tune_accuracy,
        )

        if fine_tune_accuracy is not None:
            report.transfer_gain = fine_tune_accuracy - probe_acc
            logger.info(
                f"Transfer gain (fine-tune - probe): {report.transfer_gain:.4f}"
            )

        return report

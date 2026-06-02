"""
Medical Image Style Transfer for Cross-Site Harmonization.

Implements appearance normalization techniques that reduce domain shift caused
by differences in scanner hardware, acquisition protocols, and reconstruction
algorithms across imaging sites -- without altering diagnostic content.

Key approaches:
- Fourier Domain Adaptation (FDA): swap low-frequency spectral components
- Histogram matching: align intensity distributions
- Batch normalization statistics adaptation (AdaBN)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fourier Domain Adaptation
# ---------------------------------------------------------------------------

class FourierDomainAdaptation(nn.Module):
    """Fourier-based domain adaptation (FDA).

    Swaps the low-frequency components of a source image's amplitude spectrum
    with those from a target image, effectively transferring the "style"
    (contrast, brightness, color tone) while preserving structural content
    encoded in phase.

    This is motivated by the observation that low-frequency amplitude encodes
    global appearance characteristics (scanner-specific), while phase encodes
    edges, shapes, and spatial layout (diagnostically relevant).

    Reference: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
    Segmentation", CVPR 2020

    Args:
        beta: Controls the size of the low-frequency window as a fraction
              of the spatial dimension. Smaller beta = more conservative
              style transfer (safer for medical images). Typical range: 0.01--0.1.
        random_beta: If True, sample beta uniformly from [beta_min, beta]
                     for each call, providing data augmentation.
        beta_min: Lower bound when random_beta is True.
    """

    def __init__(
        self,
        beta: float = 0.05,
        random_beta: bool = False,
        beta_min: float = 0.01,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.random_beta = random_beta
        self.beta_min = beta_min

    def _get_low_freq_mask(
        self, height: int, width: int, beta: float, device: torch.device
    ) -> torch.Tensor:
        """Create a binary mask selecting the low-frequency region around DC."""
        cy, cx = height // 2, width // 2
        rh = int(beta * height)
        rw = int(beta * width)
        rh = max(rh, 1)
        rw = max(rw, 1)

        mask = torch.zeros(height, width, device=device)
        mask[cy - rh : cy + rh, cx - rw : cx + rw] = 1.0
        return mask

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Transfer low-frequency appearance from target to source.

        Args:
            source: Source domain images (B, C, H, W)
            target: Target domain images (B, C, H, W) -- used for style only

        Returns:
            Adapted source images with target's low-frequency appearance
        """
        assert source.shape == target.shape, (
            f"Shape mismatch: source {source.shape} vs target {target.shape}"
        )

        beta = self.beta
        if self.random_beta and self.training:
            beta = self.beta_min + (self.beta - self.beta_min) * torch.rand(1).item()

        B, C, H, W = source.shape

        # FFT -> shift DC to center
        src_fft = torch.fft.fft2(source, dim=(-2, -1))
        tgt_fft = torch.fft.fft2(target, dim=(-2, -1))

        src_fft_shifted = torch.fft.fftshift(src_fft, dim=(-2, -1))
        tgt_fft_shifted = torch.fft.fftshift(tgt_fft, dim=(-2, -1))

        src_amp = torch.abs(src_fft_shifted)
        src_phase = torch.angle(src_fft_shifted)
        tgt_amp = torch.abs(tgt_fft_shifted)

        # Replace low-frequency amplitude
        mask = self._get_low_freq_mask(H, W, beta, source.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        adapted_amp = src_amp * (1.0 - mask) + tgt_amp * mask

        # Reconstruct with adapted amplitude and original phase
        adapted_fft = adapted_amp * torch.exp(1j * src_phase)
        adapted_fft = torch.fft.ifftshift(adapted_fft, dim=(-2, -1))
        adapted = torch.fft.ifft2(adapted_fft, dim=(-2, -1)).real

        return adapted.clamp(0, 1)


# ---------------------------------------------------------------------------
# Histogram Matching
# ---------------------------------------------------------------------------

class HistogramMatcher:
    """Match intensity histograms between imaging sites.

    Transforms the intensity distribution of a source image to match a
    reference distribution, reducing contrast and brightness differences
    across scanners.

    For multi-site studies, a reference histogram can be computed from a
    curated reference set representing the "canonical" appearance.
    """

    def __init__(self, n_bins: int = 256) -> None:
        self.n_bins = n_bins
        self._reference_cdf: Optional[np.ndarray] = None
        self._reference_bins: Optional[np.ndarray] = None

    def fit_reference(self, images: np.ndarray) -> None:
        """Compute reference CDF from a batch of target-domain images.

        Args:
            images: Array of images from the reference site (N, H, W) or (N, 1, H, W).
                    Values expected in [0, 1].
        """
        images = images.reshape(-1)
        hist, bin_edges = np.histogram(images, bins=self.n_bins, range=(0, 1), density=True)
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]  # normalize to [0, 1]
        self._reference_cdf = cdf
        self._reference_bins = bin_edges[:-1]
        logger.info(f"Fitted reference histogram from {len(images)} pixels")

    def match(self, image: np.ndarray) -> np.ndarray:
        """Match a single image's histogram to the reference.

        Args:
            image: Input image (H, W) with values in [0, 1].

        Returns:
            Histogram-matched image.
        """
        if self._reference_cdf is None:
            raise RuntimeError("Must call fit_reference() before match()")

        flat = image.ravel()
        hist, bin_edges = np.histogram(flat, bins=self.n_bins, range=(0, 1), density=True)
        src_cdf = np.cumsum(hist)
        src_cdf = src_cdf / src_cdf[-1]

        # For each source intensity, find the reference intensity with the closest CDF value
        mapping = np.interp(src_cdf, self._reference_cdf, self._reference_bins)

        # Digitize source pixels and apply mapping
        indices = np.clip(
            np.digitize(flat, bin_edges[:-1]) - 1, 0, self.n_bins - 1
        )
        matched = mapping[indices].reshape(image.shape)
        return matched.astype(np.float32)

    def match_batch_torch(
        self, images: torch.Tensor, reference_images: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable approximate histogram matching using soft sorting.

        This is a simplified, approximately differentiable version for use
        during training. For exact matching, use the numpy-based `match()`.

        Args:
            images: Source images (B, 1, H, W)
            reference_images: Target images (B, 1, H, W)

        Returns:
            Approximately matched images.
        """
        B, C, H, W = images.shape
        results = []

        for i in range(B):
            src = images[i].reshape(-1)
            ref_idx = i % reference_images.size(0)
            ref = reference_images[ref_idx].reshape(-1)

            # Sort both; map source ranks to reference values
            src_sorted, src_indices = torch.sort(src)
            ref_sorted, _ = torch.sort(ref)

            # If sizes differ, interpolate reference to match source length
            if len(ref_sorted) != len(src_sorted):
                ref_sorted = F.interpolate(
                    ref_sorted.unsqueeze(0).unsqueeze(0),
                    size=len(src_sorted),
                    mode="linear",
                    align_corners=False,
                ).squeeze()

            # Scatter reference values back to source positions
            matched = torch.empty_like(src)
            matched[src_indices] = ref_sorted
            results.append(matched.reshape(C, H, W))

        return torch.stack(results)


# ---------------------------------------------------------------------------
# Adaptive Batch Normalization (AdaBN)
# ---------------------------------------------------------------------------

class AdaptiveBatchNorm(nn.Module):
    """Adaptive Batch Normalization for domain transfer.

    Replaces the source-domain running statistics in BatchNorm layers
    with statistics computed on the target domain. This is a zero-cost
    adaptation that requires no additional training -- just a forward
    pass over target data to accumulate statistics.

    Reference: Li et al., "Revisiting Batch Normalization for Practical
    Domain Adaptation", ICLR Workshop 2017
    """

    @staticmethod
    def collect_bn_layers(model: nn.Module) -> List[nn.BatchNorm2d]:
        """Find all BatchNorm2d layers in a model."""
        bn_layers = []
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_layers.append(module)
        return bn_layers

    @staticmethod
    @torch.no_grad()
    def adapt(
        model: nn.Module,
        target_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: int = 100,
    ) -> None:
        """Adapt BatchNorm statistics to target domain.

        Runs the model in eval mode with tracking enabled on target data
        to replace running_mean and running_var with target statistics.

        Args:
            model: Model with BatchNorm layers to adapt.
            target_loader: DataLoader for target domain images.
            device: Device to run on.
            max_batches: Maximum number of batches for statistics estimation.
        """
        bn_layers = AdaptiveBatchNorm.collect_bn_layers(model)
        if not bn_layers:
            logger.warning("No BatchNorm layers found in model")
            return

        # Reset running stats
        for bn in bn_layers:
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.num_batches_tracked.zero_()
            bn.momentum = None  # Use cumulative moving average

        model.train()  # Enable BN stats tracking

        for i, batch in enumerate(target_loader):
            if i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            model(images)

        model.eval()
        logger.info(
            f"Adapted {len(bn_layers)} BN layers using {min(i + 1, max_batches)} target batches"
        )


# ---------------------------------------------------------------------------
# Multi-Site Style Harmonizer
# ---------------------------------------------------------------------------

class MultiSiteHarmonizer(nn.Module):
    """Combines multiple style transfer techniques for multi-site harmonization.

    Applies a configurable pipeline of appearance normalization steps to
    reduce scanner- and protocol-induced domain shift while preserving
    diagnostically relevant image content.
    """

    def __init__(
        self,
        use_fda: bool = True,
        fda_beta: float = 0.05,
        use_histogram: bool = True,
        hist_n_bins: int = 256,
        random_augment: bool = True,
    ) -> None:
        super().__init__()
        self.random_augment = random_augment

        self.fda: Optional[FourierDomainAdaptation] = None
        if use_fda:
            self.fda = FourierDomainAdaptation(
                beta=fda_beta, random_beta=random_augment, beta_min=0.01
            )

        self.histogram_matcher: Optional[HistogramMatcher] = None
        if use_histogram:
            self.histogram_matcher = HistogramMatcher(hist_n_bins)

        # Learnable per-site normalization (optional, requires site labels)
        self.site_norms: Optional[nn.ModuleDict] = None

    def register_sites(self, site_names: List[str], n_channels: int = 1) -> None:
        """Register site-specific normalization layers."""
        self.site_norms = nn.ModuleDict({
            name: nn.InstanceNorm2d(n_channels, affine=True)
            for name in site_names
        })

    def forward(
        self,
        images: torch.Tensor,
        reference_images: Optional[torch.Tensor] = None,
        site_label: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply harmonization pipeline.

        Args:
            images: Input images (B, C, H, W)
            reference_images: Reference-site images for FDA/histogram matching
            site_label: Site identifier for site-specific normalization

        Returns:
            Harmonized images
        """
        x = images

        # Fourier domain adaptation (if reference images provided)
        if self.fda is not None and reference_images is not None:
            x = self.fda(x, reference_images)

        # Site-specific normalization
        if self.site_norms is not None and site_label is not None:
            if site_label in self.site_norms:
                x = self.site_norms[site_label](x)

        return x

    def harmonize_numpy(
        self,
        image: np.ndarray,
        reference_images: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Numpy-based harmonization for inference (histogram matching).

        Args:
            image: Single image (H, W) in [0, 1].
            reference_images: Reference images to compute target histogram from.

        Returns:
            Harmonized image.
        """
        if self.histogram_matcher is not None and reference_images is not None:
            if self.histogram_matcher._reference_cdf is None:
                self.histogram_matcher.fit_reference(reference_images)
            return self.histogram_matcher.match(image)
        return image

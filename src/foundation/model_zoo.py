"""
Foundation Model Zoo: Load and adapt pretrained models for medical imaging.

Supports architecture-specific weight adaptation strategies for transitioning
from natural-image pretrained weights to single-channel, high-resolution
medical image inputs. Handles channel adaptation, resolution scaling, and
feature pyramid extraction for downstream clinical tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

logger = logging.getLogger(__name__)


class BackboneFamily(Enum):
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    VIT = "vit"
    CONVNEXT = "convnext"
    SWIN = "swin"


@dataclass
class ModelSpec:
    """Specification for a foundation model backbone."""

    name: str
    family: BackboneFamily
    input_channels: int = 3
    default_resolution: int = 224
    pretrained_weights: Optional[str] = None
    feature_dims: List[int] = field(default_factory=list)
    num_stages: int = 4


# Registry of supported architectures with their specifications
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "resnet50": ModelSpec(
        name="resnet50",
        family=BackboneFamily.RESNET,
        feature_dims=[256, 512, 1024, 2048],
        pretrained_weights="IMAGENET1K_V2",
    ),
    "resnet101": ModelSpec(
        name="resnet101",
        family=BackboneFamily.RESNET,
        feature_dims=[256, 512, 1024, 2048],
        pretrained_weights="IMAGENET1K_V2",
    ),
    "efficientnet_b0": ModelSpec(
        name="efficientnet_b0",
        family=BackboneFamily.EFFICIENTNET,
        feature_dims=[24, 40, 112, 320],
        pretrained_weights="IMAGENET1K_V1",
    ),
    "efficientnet_b4": ModelSpec(
        name="efficientnet_b4",
        family=BackboneFamily.EFFICIENTNET,
        default_resolution=380,
        feature_dims=[32, 56, 160, 448],
        pretrained_weights="IMAGENET1K_V1",
    ),
    "vit_base": ModelSpec(
        name="vit_b_16",
        family=BackboneFamily.VIT,
        feature_dims=[768, 768, 768, 768],
        pretrained_weights="IMAGENET1K_V1",
    ),
    "vit_large": ModelSpec(
        name="vit_l_16",
        family=BackboneFamily.VIT,
        feature_dims=[1024, 1024, 1024, 1024],
        pretrained_weights="IMAGENET1K_V1",
    ),
    "convnext_base": ModelSpec(
        name="convnext_base",
        family=BackboneFamily.CONVNEXT,
        feature_dims=[128, 256, 512, 1024],
        pretrained_weights="IMAGENET1K_V1",
    ),
    "swin_base": ModelSpec(
        name="swin_b",
        family=BackboneFamily.SWIN,
        feature_dims=[128, 256, 512, 1024],
        pretrained_weights="IMAGENET1K_V1",
    ),
    "swin_v2_base": ModelSpec(
        name="swin_v2_b",
        family=BackboneFamily.SWIN,
        feature_dims=[128, 256, 512, 1024],
        pretrained_weights="IMAGENET1K_V1",
    ),
}


class ChannelAdapter(nn.Module):
    """Adapt pretrained 3-channel weights to single-channel medical images.

    Three strategies:
    - replicate: Copy grayscale input to all 3 channels (preserves original weights exactly)
    - average: Average RGB kernel weights into a single-channel kernel
    - learned: Initialize from averaged weights, then learn a 1->3 channel mapping
    """

    def __init__(
        self,
        original_conv: nn.Conv2d,
        strategy: str = "average",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        assert strategy in ("replicate", "average", "learned"), (
            f"Unknown channel adaptation strategy: {strategy}"
        )

        if strategy == "replicate":
            # Keep original conv unchanged; we'll replicate input channels in forward
            self.conv = original_conv
        elif strategy == "average":
            # Create new conv with 1 input channel, weights averaged from RGB
            self.conv = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            with torch.no_grad():
                self.conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
                if original_conv.bias is not None:
                    self.conv.bias.copy_(original_conv.bias)
        elif strategy == "learned":
            # Learnable 1->3 channel projection followed by original conv
            self.channel_proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            nn.init.constant_(self.channel_proj.weight, 1.0 / 3.0)
            self.conv = original_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.strategy == "replicate":
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return self.conv(x)
        elif self.strategy == "average":
            return self.conv(x)
        elif self.strategy == "learned":
            if x.shape[1] == 1:
                x = self.channel_proj(x)
            return self.conv(x)
        raise RuntimeError(f"Unreachable: strategy={self.strategy}")


class ResolutionAdapter(nn.Module):
    """Handle resolution mismatch between pretrained and target domains.

    Medical images often have much higher native resolution than ImageNet's 224x224.
    This module supports:
    - Direct interpolation (simple but loses detail)
    - Progressive patch extraction with position embeddings (for ViTs)
    - Multi-scale feature aggregation
    """

    def __init__(
        self,
        target_resolution: int,
        pretrained_resolution: int = 224,
        mode: str = "interpolate",
    ) -> None:
        super().__init__()
        self.target_resolution = target_resolution
        self.pretrained_resolution = pretrained_resolution
        self.mode = mode
        self.scale_factor = target_resolution / pretrained_resolution

        if mode == "multi_scale":
            self.scales = [0.5, 1.0, 2.0]
            self.scale_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "interpolate":
            if x.shape[-1] != self.pretrained_resolution:
                x = F.interpolate(
                    x,
                    size=(self.pretrained_resolution, self.pretrained_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            return x
        elif self.mode == "multi_scale":
            return self._multi_scale_forward(x)
        return x

    def _multi_scale_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features at multiple scales and aggregate."""
        outputs = []
        weights = F.softmax(self.scale_weights, dim=0)
        for i, scale in enumerate(self.scales):
            target_size = int(self.pretrained_resolution * scale)
            scaled = F.interpolate(
                x, size=(target_size, target_size), mode="bilinear", align_corners=False
            )
            # Resize back to pretrained resolution for consistent feature sizes
            if target_size != self.pretrained_resolution:
                scaled = F.interpolate(
                    scaled,
                    size=(self.pretrained_resolution, self.pretrained_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(scaled * weights[i])
        return sum(outputs)


class FeaturePyramidExtractor(nn.Module):
    """Extract multi-scale feature maps from a backbone for dense prediction tasks.

    Creates a Feature Pyramid Network (FPN) on top of backbone stage outputs,
    enabling both classification (via global pooling) and dense prediction
    (segmentation, detection) from the same adapted backbone.
    """

    def __init__(self, feature_dims: List[int], output_dim: int = 256) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for dim in feature_dims:
            self.lateral_convs.append(
                nn.Conv2d(dim, output_dim, kernel_size=1, bias=False)
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(features) == len(self.lateral_convs)

        # Top-down pathway with lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i] = laterals[i] + upsampled

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class FoundationModelWrapper(nn.Module):
    """Unified wrapper around any supported backbone with medical imaging adaptations.

    Handles channel adaptation, resolution scaling, and feature extraction
    in a single consistent interface regardless of the underlying architecture.
    """

    def __init__(
        self,
        backbone_name: str,
        input_channels: int = 1,
        target_resolution: int = 512,
        channel_strategy: str = "average",
        use_fpn: bool = False,
        fpn_output_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.spec = MODEL_REGISTRY[backbone_name]
        self.backbone_name = backbone_name
        self.use_fpn = use_fpn

        # Load pretrained backbone
        self.backbone = self._load_backbone(pretrained)

        # Adapt input channels if needed
        if input_channels != 3:
            self._adapt_input_channels(channel_strategy)

        # Resolution adapter
        self.resolution_adapter = ResolutionAdapter(
            target_resolution=target_resolution,
            pretrained_resolution=self.spec.default_resolution,
        )

        # Optional FPN for dense prediction
        if use_fpn:
            self.fpn = FeaturePyramidExtractor(self.spec.feature_dims, fpn_output_dim)

        # Classification head (replaceable)
        final_dim = self.spec.feature_dims[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Identity()  # placeholder; set via set_classifier()

        logger.info(
            f"Initialized {backbone_name} | channels={input_channels} | "
            f"resolution={target_resolution} | channel_strategy={channel_strategy} | "
            f"fpn={use_fpn}"
        )

    def _load_backbone(self, pretrained: bool) -> nn.Module:
        """Load a torchvision backbone with optional pretrained weights."""
        weights = self.spec.pretrained_weights if pretrained else None
        loader = getattr(tvm, self.spec.name, None)
        if loader is None:
            raise ValueError(f"torchvision has no model named '{self.spec.name}'")
        model = loader(weights=weights)
        return model

    def _adapt_input_channels(self, strategy: str) -> None:
        """Replace the first convolutional layer with a channel-adapted version."""
        family = self.spec.family

        if family == BackboneFamily.RESNET:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = ChannelAdapter(original_conv, strategy)
        elif family == BackboneFamily.EFFICIENTNET:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = ChannelAdapter(original_conv, strategy)
        elif family == BackboneFamily.CONVNEXT:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = ChannelAdapter(original_conv, strategy)
        elif family in (BackboneFamily.VIT, BackboneFamily.SWIN):
            # For transformers, adapt the patch embedding projection
            if hasattr(self.backbone, "conv_proj"):
                original_conv = self.backbone.conv_proj
                self.backbone.conv_proj = ChannelAdapter(original_conv, strategy)
            elif hasattr(self.backbone, "features") and hasattr(
                self.backbone.features[0][0], "weight"
            ):
                original_conv = self.backbone.features[0][0]
                self.backbone.features[0][0] = ChannelAdapter(original_conv, strategy)
        else:
            logger.warning(f"Channel adaptation not implemented for {family.value}")

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-stage feature maps from the backbone.

        Returns a list of tensors, one per backbone stage (typically 4 stages).
        """
        family = self.spec.family
        features = []

        if family == BackboneFamily.RESNET:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            for i, layer in enumerate(
                [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
            ):
                x = layer(x)
                features.append(x)
        elif family in (BackboneFamily.EFFICIENTNET, BackboneFamily.CONVNEXT, BackboneFamily.SWIN):
            # torchvision ConvNeXt/EfficientNet/Swin use sequential .features
            stage_indices = self._get_stage_indices()
            for start, end in stage_indices:
                for j in range(start, end):
                    x = self.backbone.features[j](x)
                features.append(x)
        elif family == BackboneFamily.VIT:
            # ViT doesn't have natural stages; split transformer blocks evenly
            x = self._vit_patch_embed(x)
            blocks = self.backbone.encoder.layers
            n_blocks = len(blocks)
            blocks_per_stage = n_blocks // self.spec.num_stages
            for stage in range(self.spec.num_stages):
                start = stage * blocks_per_stage
                end = start + blocks_per_stage if stage < self.spec.num_stages - 1 else n_blocks
                for j in range(start, end):
                    x = blocks[j](x)
                features.append(x)

        return features

    def _get_stage_indices(self) -> List[Tuple[int, int]]:
        """Get start/end indices for each stage in sequential backbone features."""
        n = len(self.backbone.features)
        if self.spec.family == BackboneFamily.EFFICIENTNET:
            # EfficientNet stage boundaries (approximate)
            return [(0, 2), (2, 4), (4, 6), (6, n)]
        elif self.spec.family == BackboneFamily.CONVNEXT:
            return [(0, 2), (2, 4), (4, 6), (6, n)]
        elif self.spec.family == BackboneFamily.SWIN:
            return [(0, 2), (2, 4), (4, 6), (6, n)]
        return [(0, n)]

    def _vit_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Run ViT patch embedding and prepend CLS token."""
        x = self.backbone.conv_proj(x)
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        return x

    def set_classifier(self, num_classes: int, dropout: float = 0.1) -> None:
        """Replace the classification head."""
        final_dim = self.spec.feature_dims[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.resolution_adapter(x)
        features = self.extract_features(x)

        if self.use_fpn:
            pyramid = self.fpn(features)
            if return_features:
                return pyramid, features

        # Global average pooling on the last stage
        pooled = self.pool(features[-1]).flatten(1)
        logits = self.classifier(pooled)

        if return_features:
            return logits, features
        return logits

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        """Return parameter groups ordered from lowest to highest layer.

        Useful for layer-wise learning rate decay during fine-tuning.
        """
        family = self.spec.family
        groups: List[List[nn.Parameter]] = []

        if family == BackboneFamily.RESNET:
            groups.append(list(self.backbone.conv1.parameters()) + list(self.backbone.bn1.parameters()))
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                groups.append(list(getattr(self.backbone, layer_name).parameters()))
        elif family in (BackboneFamily.EFFICIENTNET, BackboneFamily.CONVNEXT, BackboneFamily.SWIN):
            for stage_start, stage_end in self._get_stage_indices():
                params = []
                for j in range(stage_start, stage_end):
                    params.extend(self.backbone.features[j].parameters())
                groups.append(params)
        elif family == BackboneFamily.VIT:
            groups.append(list(self.backbone.conv_proj.parameters()))
            blocks = list(self.backbone.encoder.layers)
            blocks_per_group = max(1, len(blocks) // 4)
            for i in range(0, len(blocks), blocks_per_group):
                params = []
                for b in blocks[i : i + blocks_per_group]:
                    params.extend(b.parameters())
                groups.append(params)

        # Always add classifier head as the last (highest LR) group
        groups.append(list(self.classifier.parameters()))
        return groups

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for linear probing)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze all backbone parameters")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze all backbone parameters")


def build_foundation_model(
    backbone: str = "swin_base",
    input_channels: int = 1,
    num_classes: int = 2,
    target_resolution: int = 512,
    channel_strategy: str = "average",
    use_fpn: bool = False,
    pretrained: bool = True,
) -> FoundationModelWrapper:
    """Factory function to build a complete foundation model for medical imaging."""
    model = FoundationModelWrapper(
        backbone_name=backbone,
        input_channels=input_channels,
        target_resolution=target_resolution,
        channel_strategy=channel_strategy,
        use_fpn=use_fpn,
        pretrained=pretrained,
    )
    model.set_classifier(num_classes)
    return model

"""AI01 dense masked autoencoder: HF Swin V2 Tiny + hierarchical V2 decoder.

Architecture: Liu et al., arXiv:2111.09883v2, sections 3.2/3.3/3.6, and
Microsoft's models/swin_transformer_v2.py. Masking/decoder/objective are an
AI01 Swin-MAE adaptation, NOT the paper's SimMIM pretraining recipe.
"""

import math

import torch
from torch import nn
from transformers import Swinv2Backbone, Swinv2Config
from transformers.models.swinv2.modeling_swinv2 import Swinv2Layer, Swinv2Stage

from utils.ai01_layers import PatchExpansion
from utils.ai01_targets import DEFAULT_TARGET_Z, validate_target_z


STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]
TRAINING_FORMAT = "ai01-swinv2-mae-training-v1"


def ai01_v2_swin_config(image_size=256, window_size=8, drop_path_rate=0.2):
    """SwinV2-T (2,2,6,2), with all 11 Z as input; no network downloads."""
    return Swinv2Config(
        image_size=image_size, patch_size=4, num_channels=11, embed_dim=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=window_size,
        mlp_ratio=4.0, qkv_bias=True, hidden_act="gelu", layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        drop_path_rate=drop_path_rate, use_absolute_embeddings=False,
        pretrained_window_sizes=[0, 0, 0, 0], out_features=STAGE_NAMES,
    )


def initialize_residual_post_norm(module):
    """Match Microsoft's BasicLayer._init_respostnorm, after HF initialization."""
    for block in module.modules():
        if isinstance(block, Swinv2Layer):
            for norm in (block.layernorm_before, block.layernorm_after):
                nn.init.zeros_(norm.weight)
                nn.init.zeros_(norm.bias)


class AI01V2ReconstructionDecoder(nn.Module):
    """Upsample 32->16->8->4 strides; fuse masked encoder stage features."""

    def __init__(self, config, depth=2):
        super().__init__()
        self.expansions = nn.ModuleList()
        self.stages = nn.ModuleList()
        for index in (2, 1, 0):
            dim = config.embed_dim * 2 ** index
            grid = config.image_size // config.patch_size // 2 ** index
            self.expansions.append(PatchExpansion(2 * dim))
            self.stages.append(Swinv2Stage(
                config, dim=dim, input_resolution=(grid, grid), depth=depth,
                num_heads=config.num_heads[index], drop_path=[0.0] * depth,
                downsample=None, pretrained_window_size=config.pretrained_window_sizes[index],
            ))
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        # Keep the 11-Z patch layout; loss selects target_z, not input channels.
        self.prediction = nn.Linear(config.embed_dim, config.patch_size ** 2 * config.num_channels)
        initialize_residual_post_norm(self)

    def forward(self, features):
        x = features[-1].permute(0, 2, 3, 1).contiguous()
        for expand, stage, skip in zip(self.expansions, self.stages, reversed(features[:-1])):
            x = (expand(x) + skip.permute(0, 2, 3, 1)) / math.sqrt(2.0)
            batch, height, width, channels = x.shape
            x = stage(x.reshape(batch, height * width, channels), (height, width))[0]
            x = x.reshape(batch, height, width, channels)
        return self.prediction(self.norm(x.flatten(1, 2)))


class AI01V2SwinMAE(nn.Module):
    def __init__(self, mask_ratio=0.75, mask_group_size=4, norm_pix_loss=False,
                 decoder_depth=2, encoder_config=None, image_size=None,
                 target_z=DEFAULT_TARGET_Z):
        super().__init__()
        if isinstance(encoder_config, dict):
            if encoder_config.get("model_type") != "swinv2":
                raise ValueError("encoder_config must describe Swin V2, not V1")
            encoder_config = Swinv2Config.from_dict(encoder_config)
        self.config = encoder_config if encoder_config is not None else ai01_v2_swin_config(
            image_size=256 if image_size is None else image_size,
        )
        if not isinstance(self.config, Swinv2Config):
            raise TypeError("AI01 V2 requires Swinv2Config")
        if image_size is not None and image_size != self.config.image_size:
            raise ValueError("image_size and encoder_config.image_size must agree")
        size = self.config.image_size
        if not isinstance(size, int) or isinstance(size, bool) or size < 32 or size % 32:
            raise ValueError("image_size must be a positive integer multiple of 32")
        if self.config.patch_size != 4 or self.config.num_channels != 11:
            raise ValueError("AI01 V2 requires patch_size=4 and all 11 input Z channels")
        if len(self.config.depths) != 4 or len(self.config.num_heads) != 4:
            raise ValueError("AI01 V2 requires four encoder stages")
        if any(depth < 1 for depth in self.config.depths):
            raise ValueError("Every encoder stage must have at least one block")
        if list(self.config.out_features) != STAGE_NAMES or self.config.use_absolute_embeddings:
            raise ValueError("Require all four output stages and no absolute embeddings")
        if not isinstance(self.config.window_size, int) or self.config.window_size < 2:
            raise ValueError("window_size must be an integer >= 2")
        if not 0 <= self.config.drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        if mask_group_size < 1 or decoder_depth < 1:
            raise ValueError("mask_group_size and decoder_depth must be positive")
        grid = size // self.config.patch_size
        if grid % mask_group_size or (grid // mask_group_size) ** 2 < 2:
            raise ValueError("Mask groups must tile the patch grid, with at least two groups")
        self.mask_ratio, self.mask_group_size = mask_ratio, mask_group_size
        self.norm_pix_loss, self.decoder_depth = norm_pix_loss, decoder_depth
        self.target_z = validate_target_z(target_z, self.config.num_channels)
        self.backbone = Swinv2Backbone(self.config)
        initialize_residual_post_norm(self.backbone)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = AI01V2ReconstructionDecoder(self.config, decoder_depth)

    def training_config(self):
        # Save the entire backbone spec, including window/pretrained window sizes.
        return {
            "image_size": self.config.image_size, "mask_ratio": self.mask_ratio,
            "mask_group_size": self.mask_group_size, "norm_pix_loss": self.norm_pix_loss,
            "decoder_depth": self.decoder_depth, "target_z": list(self.target_z),
            "encoder_config": self.config.to_dict(),
        }

    def make_mask(self, batch_size, device, generator=None):
        grid, group = self.config.image_size // self.config.patch_size, self.mask_group_size
        coarse = grid // group
        count = coarse ** 2
        num_masked = min(count - 1, max(1, round(count * self.mask_ratio)))
        order = torch.rand(batch_size, count, device=device, generator=generator).argsort(dim=1)
        mask = torch.zeros(batch_size, count, device=device, dtype=torch.bool)
        mask.scatter_(1, order[:, :num_masked], True)
        return mask.reshape(batch_size, coarse, coarse).repeat_interleave(
            group, 1,
        ).repeat_interleave(group, 2).flatten(1)

    def _validate_images(self, images):
        expected = (11, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected [B, {expected}], got {tuple(images.shape)}")

    def patchify(self, images):
        self._validate_images(images)
        patch = self.config.patch_size
        batch, channels, height, width = images.shape
        x = images.reshape(batch, channels, height // patch, patch, width // patch, patch)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch * patch * channels)

    def unpatchify(self, patches):
        patch, channels = self.config.patch_size, self.config.num_channels
        grid = self.config.image_size // patch
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (grid * grid, patch * patch * channels):
            raise ValueError("Invalid reconstruction patch shape")
        x = patches.reshape(patches.shape[0], grid, grid, patch, patch, channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(patches.shape[0], channels, grid * patch, grid * patch)

    def forward_encoder(self, images, mask):
        self._validate_images(images)
        embeddings, dimensions = self.backbone.embeddings(images)
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(embeddings.shape[:2]):
            raise ValueError("Expected boolean patch mask [B, (image_size/4)^2]")
        embeddings = torch.where(mask.unsqueeze(-1), self.mask_token.to(embeddings.dtype), embeddings)
        outputs = self.backbone.encoder(
            embeddings, dimensions, output_hidden_states=True,
            output_hidden_states_before_downsampling=True, return_dict=True,
        )
        # Swinv2Backbone returns stage outputs directly: unlike V1 it does NOT
        # own hidden_states_norms, nor accept always_partition in its encoder.
        return tuple(hidden for name, hidden in zip(
            self.backbone.stage_names, outputs.reshaped_hidden_states,
        ) if name in self.backbone.out_features)

    def forward_loss(self, images, prediction, mask):
        target = self.patchify(images).float()
        if prediction.shape != target.shape or mask.shape != target.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Prediction, target and boolean mask shapes must agree")
        if not mask.any():
            raise ValueError("At least one patch must be masked")
        shape = (*target.shape[:2], self.config.patch_size ** 2, self.config.num_channels)
        target = target.reshape(shape)[..., list(self.target_z)].flatten(-2)
        prediction = prediction.float().reshape(shape)[..., list(self.target_z)].flatten(-2)
        if self.norm_pix_loss:
            mean = target.mean(-1, keepdim=True)
            variance = target.var(-1, keepdim=True, unbiased=False)
            target = (target - mean) / (variance + 1e-6).sqrt()
        error = (prediction - target).square().mean(-1)
        return (error * mask).sum() / mask.sum()

    def forward(self, images, mask=None):
        if mask is None:
            mask = self.make_mask(images.shape[0], images.device)
        prediction = self.decoder(self.forward_encoder(images, mask))
        return self.forward_loss(images, prediction, mask), prediction, mask


def checkpoint_model_config(saved):
    """V2 artifacts have no legacy all-Z fallback; require an explicit full spec."""
    required = {"image_size", "mask_ratio", "mask_group_size", "norm_pix_loss",
                "decoder_depth", "target_z", "encoder_config"}
    if set(saved) != required or saved["encoder_config"].get("model_type") != "swinv2":
        raise ValueError("Invalid AI01 Swin V2 MAE model configuration")
    return {**saved, "target_z": list(validate_target_z(saved["target_z"], 11))}

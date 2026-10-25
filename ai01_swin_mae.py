"""AI01 masked autoencoder with the *same* HF SwinBackbone used by Mask2Former.

The hierarchical reconstruction decoder is adapted from the repository's Swin MAE:
patch expansion + Swin blocks, with masked multi-scale fusion to train every
backbone output normalization. It is not a reproduction of the paper's decoder.
The exported encoder has no SSL-only parameters or custom forward methods.
"""

import math

import torch
from torch import nn
from transformers import SwinBackbone, SwinConfig
from transformers.models.swin.modeling_swin import SwinStage


STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]


def ai01_swin_config():
    # Mirrors facebook/mask2former-swin-tiny-coco-instance, with 11-channel input.
    # No Hub download or pretrained RGB weights are required to start SSL.
    return SwinConfig(
        image_size=256, patch_size=4, num_channels=11, embed_dim=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
        mlp_ratio=4.0, qkv_bias=True, hidden_act="gelu", layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        drop_path_rate=0.3, use_absolute_embeddings=False,
        out_features=STAGE_NAMES,
    )


class PatchExpansion(nn.Module):
    """Double spatial resolution and halve channels (the original decoder layout)."""

    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        batch, height, width, channels = x.shape
        x = self.expand(x).reshape(batch, height, width, 2, 2, channels // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, 2 * height, 2 * width, channels // 2)
        return self.norm(x)


class ReconstructionDecoder(nn.Module):
    def __init__(self, config, depth=2):
        super().__init__()
        self.expansions = nn.ModuleList()
        self.stages = nn.ModuleList()
        for index in (2, 1, 0):
            dim = config.embed_dim * 2 ** index
            self.expansions.append(PatchExpansion(2 * dim))
            grid = config.image_size // config.patch_size // 2 ** index
            self.stages.append(SwinStage(
                config, dim=dim, input_resolution=(grid, grid), depth=depth,
                num_heads=config.num_heads[index], drop_path=[0.0] * depth,
                downsample=None,
            ))
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.prediction = nn.Linear(config.embed_dim, config.patch_size ** 2 * config.num_channels)

    def forward(self, feature_maps):
        x = feature_maps[-1].permute(0, 2, 3, 1).contiguous()
        for expand, stage, skip in zip(self.expansions, self.stages, reversed(feature_maps[:-1])):
            x = expand(x)
            x = (x + skip.permute(0, 2, 3, 1)) / math.sqrt(2.0)
            batch, height, width, channels = x.shape
            # HF handles padding for 64x64/window7, unlike the original RGB decoder.
            x = stage(x.reshape(batch, height * width, channels), (height, width),
                      always_partition=True)[0]
            x = x.reshape(batch, height, width, channels)
        return self.prediction(self.norm(x.flatten(1, 2)))


class AI01SwinMAE(nn.Module):
    def __init__(self, mask_ratio=0.75, mask_group_size=4, norm_pix_loss=False,
                 decoder_depth=2, encoder_config=None):
        super().__init__()
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        if mask_group_size < 1 or decoder_depth < 1:
            raise ValueError("mask_group_size and decoder_depth must be positive")
        self.config = encoder_config if encoder_config is not None else ai01_swin_config()
        grid = self.config.image_size // self.config.patch_size
        if self.config.image_size % (self.config.patch_size * 8):
            raise ValueError("Image size must be divisible by the four-stage encoder stride (32)")
        if grid % mask_group_size or (grid // mask_group_size) ** 2 < 2:
            raise ValueError("Mask groups must tile the patch grid, with at least two groups")
        if list(self.config.out_features) != STAGE_NAMES:
            raise ValueError("All four backbone output stages are required")
        self.mask_ratio = mask_ratio
        self.mask_group_size = mask_group_size
        self.norm_pix_loss = norm_pix_loss
        self.decoder_depth = decoder_depth
        self.backbone = SwinBackbone(self.config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = ReconstructionDecoder(self.config, depth=decoder_depth)

    def training_config(self):
        return {
            "mask_ratio": self.mask_ratio,
            "mask_group_size": self.mask_group_size,
            "norm_pix_loss": self.norm_pix_loss,
            "decoder_depth": self.decoder_depth,
        }

    def make_mask(self, batch_size, device, generator=None):
        grid = self.config.image_size // self.config.patch_size
        group = self.mask_group_size
        coarse = grid // group
        count = coarse ** 2
        num_masked = min(count - 1, max(1, round(count * self.mask_ratio)))
        order = torch.rand(batch_size, count, device=device, generator=generator).argsort(dim=1)
        mask = torch.zeros(batch_size, count, device=device, dtype=torch.bool)
        mask.scatter_(1, order[:, :num_masked], True)
        return mask.reshape(batch_size, coarse, coarse).repeat_interleave(
            group, dim=1
        ).repeat_interleave(group, dim=2).flatten(1)

    def _validate_images(self, images):
        expected = (self.config.num_channels, self.config.image_size, self.config.image_size)
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
        expected = embeddings.shape[:2]
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(expected):
            raise ValueError(f"Expected boolean patch mask {tuple(expected)}")
        embeddings = torch.where(mask.unsqueeze(-1), self.mask_token.to(embeddings.dtype), embeddings)
        # Same forward path as SwinBackbone.forward, with masked embeddings only.
        outputs = self.backbone.encoder(
            embeddings, dimensions, output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            always_partition=True, return_dict=True,
        )
        features = []
        for name, hidden in zip(self.backbone.stage_names, outputs.reshaped_hidden_states):
            if name in self.backbone.out_features:
                hidden = hidden.permute(0, 2, 3, 1)
                hidden = self.backbone.hidden_states_norms[name](hidden)
                features.append(hidden.permute(0, 3, 1, 2).contiguous())
        return tuple(features)

    def forward_loss(self, images, prediction, mask):
        # Compute MSE in fp32 even under autocast; raw normalized pixels by default.
        target = self.patchify(images).float()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (variance + 1e-6).sqrt()
        if prediction.shape != target.shape or mask.shape != target.shape[:2]:
            raise ValueError("Prediction, target and mask shapes must agree")
        if not mask.any():
            raise ValueError("At least one patch must be masked")
        error = (prediction.float() - target).square().mean(dim=-1)
        return (error * mask).sum() / mask.sum()

    def forward(self, images, mask=None):
        if mask is None:
            mask = self.make_mask(images.shape[0], images.device)
        features = self.forward_encoder(images, mask)
        prediction = self.decoder(features)
        return self.forward_loss(images, prediction, mask), prediction, mask

"""Strict, encoder-only transfer from SSL to A1_Refractor Mask2Former."""

from pathlib import Path

import torch
import transformers
from transformers import SwinBackbone, SwinConfig

from utils.ai01_data import DEFAULT_PREPROCESSING
from utils.ai01_preprocessing import validate_preprocessing


FORMAT = "ai01-swin-backbone-v1"
# image_size is intentionally excluded: without absolute positional embeddings,
# HF's 224 config and AI01's actual 256 input use identical weights and padding.
ARCHITECTURE_FIELDS = (
    "patch_size", "num_channels", "embed_dim", "depths", "num_heads", "window_size",
    "mlp_ratio", "qkv_bias", "hidden_act", "use_absolute_embeddings", "layer_norm_eps",
    "hidden_dropout_prob", "attention_probs_dropout_prob", "drop_path_rate", "out_features",
)


def export_encoder(backbone, path, preprocessing=None):
    if not isinstance(backbone, SwinBackbone):
        raise TypeError("Export requires a transformers.SwinBackbone")
    spec = dict(DEFAULT_PREPROCESSING if preprocessing is None else preprocessing)
    if backbone.config.num_channels != spec["num_channels"]:
        raise ValueError("Encoder/preprocessing channel counts disagree")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": FORMAT,
        "transformers_version": str(transformers.__version__),
        "backbone_config": backbone.config.to_dict(),
        "preprocessing": spec,
        "encoder": {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()},
    }, temporary)
    temporary.replace(path)


def _read_export(path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") != FORMAT:
        raise ValueError("Not an AI01 encoder export; use encoder-last.pt, not the SSL training checkpoint")
    return artifact


def load_ai01_encoder(model, path, image_processor=None):
    """Load into a SwinBackbone or Mask2Former; never replace the decoder.

    Run AFTER constructing the 11-channel patch projection and BEFORE creating
    the fine-tuning optimizer. All configuration/keys/shapes are checked first.
    """
    target = model if isinstance(model, SwinBackbone) else model.model.pixel_level_module.encoder
    if not isinstance(target, SwinBackbone):
        raise TypeError("AI01 target must use transformers.SwinBackbone (not SwinV2/MambaVision)")
    artifact = _read_export(path)
    source_config = SwinConfig.from_dict(artifact["backbone_config"])
    if source_config.use_absolute_embeddings and source_config.image_size != target.config.image_size:
        raise ValueError("Different image sizes require use_absolute_embeddings=False")
    for field in ARCHITECTURE_FIELDS:
        source, destination = getattr(source_config, field), getattr(target.config, field)
        if source != destination:
            raise ValueError(f"Backbone mismatch for {field}: SSL={source}, AI01={destination}")
    validate_preprocessing(artifact["preprocessing"], image_processor)
    source_state = artifact["encoder"]
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    wrong_shapes = [key for key in set(source_state) & set(target_state)
                    if source_state[key].shape != target_state[key].shape]
    if missing or unexpected or wrong_shapes:
        raise ValueError(f"Incompatible encoder: missing={missing}, unexpected={unexpected}, shapes={wrong_shapes}")
    target.load_state_dict(source_state, strict=True)
    return {
        "loaded_tensors": len(source_state),
        "preprocessing": artifact["preprocessing"],
        "transformers_version": artifact["transformers_version"],
    }


def load_exported_backbone(path):
    """Standalone inference backbone, with ordinary HF forward -> four maps."""
    artifact = _read_export(path)
    backbone = SwinBackbone(SwinConfig.from_dict(artifact["backbone_config"]))
    backbone.load_state_dict(artifact["encoder"], strict=True)
    return backbone

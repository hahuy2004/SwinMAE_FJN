"""Strict encoder-only Swin V2 exports, standalone loading and Mask2Former use."""

from pathlib import Path

import torch
import transformers
from transformers import Swinv2Backbone, Swinv2Config

from utils.ai01_data import DEFAULT_PREPROCESSING
from utils.ai01_preprocessing import validate_preprocessing


FORMAT = "ai01-swinv2-backbone-v1"
# V2 resolves effective window/shift sizes at construction. Unlike the V1
# exporter, never silently omit image_size from compatibility checks.
ARCHITECTURE_FIELDS = (
    "image_size", "patch_size", "num_channels", "embed_dim", "depths", "num_heads",
    "window_size", "pretrained_window_sizes", "mlp_ratio", "qkv_bias", "hidden_act",
    "use_absolute_embeddings", "layer_norm_eps", "hidden_dropout_prob",
    "attention_probs_dropout_prob", "drop_path_rate", "out_features",
)


def export_encoder(backbone, path, preprocessing=None):
    if not isinstance(backbone, Swinv2Backbone):
        raise TypeError("V2 export requires transformers.Swinv2Backbone, not Swin V1")
    spec = dict(DEFAULT_PREPROCESSING if preprocessing is None else preprocessing)
    if preprocessing is None:
        spec["image_size"] = backbone.config.image_size
    validate_preprocessing(spec)
    if backbone.config.num_channels != spec["num_channels"] or backbone.config.image_size != spec["image_size"]:
        raise ValueError("Encoder/preprocessing image size or channel counts disagree")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": FORMAT, "transformers_version": str(transformers.__version__),
        "backbone_config": backbone.config.to_dict(), "preprocessing": spec,
        "encoder": {name: tensor.detach().cpu().clone() for name, tensor in backbone.state_dict().items()},
    }, temporary)
    temporary.replace(path)


def read_export(path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") != FORMAT:
        raise ValueError("Expected an AI01 Swin V2 encoder export, not a V1/full training checkpoint")
    if artifact["backbone_config"].get("model_type") != "swinv2":
        raise ValueError("Export must contain a Swin V2 configuration")
    validate_preprocessing(artifact["preprocessing"])
    config = artifact["backbone_config"]
    if config["num_channels"] != 11 or config["image_size"] != artifact["preprocessing"]["image_size"]:
        raise ValueError("Export encoder/preprocessing mismatch")
    if config["patch_size"] != 4 or config["out_features"] != ["stage1", "stage2", "stage3", "stage4"]:
        raise ValueError("Export must provide four AI01 feature stages at strides 4/8/16/32")
    return artifact


def _load_state(target, artifact, image_processor=None):
    source_config = Swinv2Config.from_dict(artifact["backbone_config"])
    for name in ARCHITECTURE_FIELDS:
        source, destination = getattr(source_config, name), getattr(target.config, name)
        if source != destination:
            raise ValueError(f"Backbone mismatch for {name}: SSL={source}, AI01={destination}")
    validate_preprocessing(artifact["preprocessing"], image_processor)
    source, destination = artifact["encoder"], target.state_dict()
    missing, unexpected = sorted(set(destination) - set(source)), sorted(set(source) - set(destination))
    wrong_shapes = sorted(name for name in set(source) & set(destination)
                          if source[name].shape != destination[name].shape)
    if missing or unexpected or wrong_shapes:
        raise ValueError(f"Incompatible V2 encoder: missing={missing}, unexpected={unexpected}, shapes={wrong_shapes}")
    # All metadata/keys/shapes are checked BEFORE any target weights are copied.
    target.load_state_dict(source, strict=True)
    return {"loaded_tensors": len(source), "preprocessing": artifact["preprocessing"],
            "transformers_version": artifact["transformers_version"], "backbone_type": "swinv2"}


def load_ai01_v2_encoder(model, path, image_processor=None):
    """Load into an existing V2 backbone/Mask2Former, before optimizer creation.

    Never replace a V1 backbone silently. The downstream model must already use
    the SAME V2 configuration and 11-channel patch projection as the export.
    """
    target = model if isinstance(model, Swinv2Backbone) else getattr(
        getattr(getattr(model, "model", None), "pixel_level_module", None), "encoder", None,
    )
    if not isinstance(target, Swinv2Backbone):
        raise TypeError("AI01 V2 target must use transformers.Swinv2Backbone (not Swin V1)")
    return _load_state(target, read_export(path), image_processor)


def load_exported_backbone(path):
    artifact = read_export(path)
    backbone = Swinv2Backbone(Swinv2Config.from_dict(artifact["backbone_config"]))
    _load_state(backbone, artifact)
    return backbone


def build_ai01_v2_mask2former(path, **task_config):
    """Fresh Mask2Former task heads + the exported SSL encoder, no downloads.

    Existing custom AI01 temporal/BiMamba heads are NOT constructed by this
    helper. Integrate those separately, using load_ai01_v2_encoder afterwards.
    """
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    forbidden = {"backbone", "backbone_config", "use_pretrained_backbone", "use_timm_backbone", "backbone_kwargs"}
    if forbidden.intersection(task_config):
        raise ValueError("Backbone settings come exclusively from the V2 encoder export")
    artifact = read_export(path)
    config = Mask2FormerConfig(backbone_config=artifact["backbone_config"], **task_config)
    model = Mask2FormerForUniversalSegmentation(config)
    _load_state(model.model.pixel_level_module.encoder, artifact)
    return model

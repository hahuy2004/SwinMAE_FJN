"""Check strict transfer and a real Mask2Former forward; no Mamba dependency."""

import argparse
import os

os.environ.setdefault("USE_TF", "0")

import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from ai01_transfer import load_ai01_encoder, load_exported_backbone


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--mask2former_config", default="facebook/mask2former-swin-tiny-coco-instance",
                        help="Hub ID or local config directory; downloads configuration, not weights")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = Mask2FormerConfig.from_pretrained(args.mask2former_config)
    config.num_labels = 2
    config.num_queries = 7
    config.backbone_config.num_channels = 11
    config.backbone_config.image_size = 256
    model = Mask2FormerForUniversalSegmentation(config)
    report = load_ai01_encoder(model, args.encoder)
    model = model.to(args.device).eval()
    reference = load_exported_backbone(args.encoder).to(args.device).eval()
    images = torch.randn(1, 11, 256, 256, device=args.device)
    expected = reference(images).feature_maps
    actual = model.model.pixel_level_module.encoder(images).feature_maps
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    result = model(pixel_values=images)
    print(report)
    print("Feature maps:", [tuple(feature.shape) for feature in actual])
    print("Class logits:", tuple(result.class_queries_logits.shape))
    print("Mask logits:", tuple(result.masks_queries_logits.shape))
    print("PASS: exact backbone outputs after strict transfer; Mask2Former forward works.")
    print("The notebook's custom BiMamba decoder still needs its own runtime/dependencies.")


if __name__ == "__main__":
    main()

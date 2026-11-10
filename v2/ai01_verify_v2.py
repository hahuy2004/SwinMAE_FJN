"""Offline verification of a V2 encoder export, optionally through Mask2Former."""

import argparse
import json
import os

os.environ.setdefault("USE_TF", "0")

import torch

from .ai01_transfer_v2 import build_ai01_v2_mask2former, load_exported_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mask2former", action="store_true")
    args = parser.parse_args()
    backbone = load_exported_backbone(args.encoder).to(args.device).eval()
    size = backbone.config.image_size
    images = torch.randn(1, 11, size, size, device=args.device)
    with torch.no_grad():
        features = backbone(images).feature_maps
        if len(features) != 4 or not all(torch.isfinite(feature).all() for feature in features):
            raise ValueError("Expected four finite feature maps")
        report = {"backbone": "swinv2", "feature_shapes": [list(x.shape) for x in features]}
        if args.mask2former:
            model = build_ai01_v2_mask2former(args.encoder, num_labels=2, num_queries=7).to(args.device).eval()
            for actual, expected in zip(model.model.pixel_level_module.encoder(images).feature_maps, features):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            outputs = model(pixel_values=images)
            if not torch.isfinite(outputs.class_queries_logits).all() or not torch.isfinite(outputs.masks_queries_logits).all():
                raise ValueError("Non-finite Mask2Former outputs")
            report.update(class_logits=list(outputs.class_queries_logits.shape),
                          mask_logits=list(outputs.masks_queries_logits.shape))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

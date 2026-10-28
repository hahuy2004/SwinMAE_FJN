"""224/256 crops, SSL checkpoint reconstruction, and 224 -> 256 transfer."""

from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from ai01_data import DEFAULT_PREPROCESSING, load_zstack
from ai01_swin_mae import AI01SwinMAE, ai01_swin_config
from ai01_transfer import export_encoder, load_ai01_encoder, validate_preprocessing
from train_ai01 import parser


class ImageSizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_defaults_choices_and_invalid_sizes(self):
        self.assertEqual(parser().parse_args(["--manifest", "unused.txt"]).image_size, 256)
        for size in (224, 256):
            args = parser().parse_args(["--manifest", "unused.txt", "--image_size", str(size)])
            self.assertEqual(args.image_size, size)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["--manifest", "unused.txt", "--image_size", "225"])

    def test_selected_size_is_a_center_crop_not_a_resize(self):
        image = np.random.default_rng(42).integers(0, 256, (11, 400, 400, 1), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.npy"
            np.save(path, image)
            for size in (224, 256):
                with self.subTest(size=size):
                    spec = {**DEFAULT_PREPROCESSING, "image_size": size}
                    actual = load_zstack(path, preprocessing=spec)
                    start = (400 - size) // 2
                    expected = image[:, start:start + size, start:start + size, 0].astype(np.float32)
                    expected = (expected * spec["rescale_factor"] - spec["mean"]) / spec["std"]
                    self.assertEqual(tuple(actual.shape), (11, size, size))
                    torch.testing.assert_close(actual, torch.from_numpy(expected))

    def test_224_forward_gradients_and_saved_model_config(self):
        model = AI01SwinMAE(image_size=224)
        images = torch.randn(1, 11, 224, 224)
        torch.testing.assert_close(model.unpatchify(model.patchify(images)), images, rtol=0, atol=0)
        loss, prediction, mask = model(images)
        self.assertEqual(tuple(prediction.shape), (1, 3136, 176))
        self.assertEqual(mask.sum().item(), 2352)
        groups = mask.reshape(1, 14, 4, 14, 4).permute(0, 1, 3, 2, 4)
        self.assertTrue(torch.equal(groups, groups[..., :1, :1].expand_as(groups)))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        restored = AI01SwinMAE(**model.training_config())
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(restored.config.image_size, 224)
        # Pre-existing full checkpoints omit image_size and still rebuild at 256.
        old_config = {key: value for key, value in model.training_config().items() if key != "image_size"}
        self.assertEqual(AI01SwinMAE(**old_config).config.image_size, 256)
        with self.assertRaisesRegex(ValueError, "must agree"):
            AI01SwinMAE(image_size=224, encoder_config=ai01_swin_config(256))

    def test_224_ssl_encoder_transfers_to_256_mask2former(self):
        source = AI01SwinMAE(image_size=224).eval()
        spec = {**DEFAULT_PREPROCESSING, "image_size": 224}
        processor = Mask2FormerImageProcessor(
            do_resize=False, image_mean=[spec["mean"]], image_std=[spec["std"]],
            rescale_factor=spec["rescale_factor"],
        )
        target = Mask2FormerForUniversalSegmentation(Mask2FormerConfig(
            backbone_config=ai01_swin_config(256).to_dict(), num_labels=2, num_queries=7,
        )).eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.pt"
            export_encoder(source.backbone, path, spec)
            report = load_ai01_encoder(target, path, image_processor=processor)
        self.assertEqual(report["loaded_tensors"], len(source.backbone.state_dict()))
        images = torch.randn(1, 11, 256, 256)
        with torch.inference_mode():
            expected = source.backbone(images).feature_maps
            actual = target.model.pixel_level_module.encoder(images).feature_maps
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            outputs = target(pixel_values=images)
        self.assertEqual(tuple(outputs.masks_queries_logits.shape), (1, 7, 64, 64))

    def test_transfer_still_checks_other_preprocessing_fields(self):
        spec = {**DEFAULT_PREPROCESSING, "image_size": 224}
        validate_preprocessing(spec)
        for key, value in (("mean", 0.0), ("std", 1.0), ("crop", "resize"),
                           ("channel_order", list(reversed(range(11))))):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_preprocessing({**spec, key: value})
        with self.assertRaisesRegex(ValueError, "image_size"):
            validate_preprocessing({**spec, "image_size": 225})

    def test_32_pixel_mask_groups_work_at_both_image_sizes(self):
        for size, groups, masked_groups in ((224, 7, 37), (256, 8, 48)):
            with self.subTest(size=size):
                model = AI01SwinMAE(image_size=size, mask_group_size=8).eval()
                mask = model.make_mask(1, "cpu")
                self.assertEqual(mask.sum().item(), masked_groups * 64)
                tiled = mask.reshape(1, groups, 8, groups, 8).permute(0, 1, 3, 2, 4)
                self.assertTrue(torch.equal(tiled, tiled[..., :1, :1].expand_as(tiled)))
                with torch.inference_mode():
                    loss, prediction, _ = model(torch.randn(1, 11, size, size), mask)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(tuple(prediction.shape), (1, (size // 4) ** 2, 176))


if __name__ == "__main__":
    unittest.main()

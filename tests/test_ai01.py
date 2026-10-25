"""Offline tests: data parity, masking, gradients, strict transfer, downstream forward."""

import json
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

from ai01_data import (
    AI01UnlabelledDataset, DEFAULT_PREPROCESSING, ensure_disjoint_wells, load_zstack,
)
from ai01_swin_mae import AI01SwinMAE, ai01_swin_config
from ai01_transfer import export_encoder, load_ai01_encoder, load_exported_backbone


class DataTests(unittest.TestCase):
    def test_crop_normalization_and_layouts_match_notebook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.arange(11, dtype=np.uint8)[:, None, None, None]
            image = np.broadcast_to(values, (11, 400, 400, 1)).copy()
            # Include spatial variation to verify crop and ordering, not just shape.
            image[:, 72:328, 72:328, 0] += np.arange(256, dtype=np.uint8)[None, :, None] // 2
            np.save(root / "frame.npy", image)
            actual = load_zstack(root / "frame.npy")
            spec = DEFAULT_PREPROCESSING
            processor = Mask2FormerImageProcessor(
                do_resize=False, image_mean=[spec["mean"]], image_std=[spec["std"]],
                rescale_factor=spec["rescale_factor"],
            )
            # A1_Refractor feeds one grayscale image per Z plane, then reshapes.
            expected = processor([plane[72:328, 72:328] for plane in image], return_tensors="pt")
            expected = expected["pixel_values"].reshape(1, 11, 256, 256)[0]
            torch.testing.assert_close(actual, expected)
            for name, array in (("zhw", image[..., 0]), ("hwz", image[..., 0].transpose(1, 2, 0))):
                np.save(root / f"{name}.npy", array)
                torch.testing.assert_close(load_zstack(root / f"{name}.npy"), actual)

    def test_manifest_bad_input_and_well_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for well in ("well_a", "well_b"):
                (root / well).mkdir()
                np.save(root / well / "frame.npy", np.zeros((11, 256, 256), dtype=np.uint8))
            (root / "train.txt").write_text("# relative paths\nwell_a/frame.npy\n", encoding="utf-8")
            (root / "val.txt").write_text("well_b/frame.npy\n", encoding="utf-8")
            train = AI01UnlabelledDataset(manifest=root / "train.txt", augment=False)
            validation = AI01UnlabelledDataset(manifest=root / "val.txt", augment=False)
            self.assertEqual(tuple(train[0].shape), (11, 256, 256))
            ensure_disjoint_wells(train, validation)
            with self.assertRaisesRegex(ValueError, "share well"):
                ensure_disjoint_wells(train, train)
            np.save(root / "mask.npy", np.zeros((256, 256), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "Z-stack"):
                load_zstack(root / "mask.npy")
            np.save(root / "bad.npy", np.full((11, 256, 256), np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "finite"):
                load_zstack(root / "bad.npy")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = AI01SwinMAE()

    def test_patch_round_trip_mask_and_unmasked_forward_parity(self):
        model = self.model.eval()
        images = torch.randn(1, 11, 256, 256)
        patches = model.patchify(images)
        self.assertEqual(tuple(patches.shape), (1, 4096, 176))
        torch.testing.assert_close(model.unpatchify(patches), images, rtol=0, atol=0)
        mask = model.make_mask(2, "cpu")
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(torch.equal(mask.sum(1), torch.tensor([3072, 3072])))
        grouped = mask.reshape(2, 16, 4, 16, 4).permute(0, 1, 3, 2, 4)
        self.assertTrue(torch.equal(grouped, grouped[..., :1, :1].expand_as(grouped)))
        with torch.no_grad():
            masked_path = model.forward_encoder(images, torch.zeros(1, 4096, dtype=torch.bool))
            normal_path = model.backbone(images).feature_maps
        self.assertEqual([tuple(x.shape) for x in normal_path], [
            (1, 96, 64, 64), (1, 192, 32, 32), (1, 384, 16, 16), (1, 768, 8, 8),
        ])
        for actual, expected in zip(masked_path, normal_path):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_no_masked_pixel_leakage_and_masked_only_loss(self):
        model = self.model.eval()
        image = torch.randn(1, 11, 256, 256)
        mask = model.make_mask(1, "cpu")
        image_mask = mask.reshape(1, 1, 64, 64).repeat_interleave(4, -2).repeat_interleave(4, -1)
        changed = torch.where(image_mask, torch.full_like(image, 100.0), image)
        with torch.no_grad():
            features = model.forward_encoder(image, mask)
            changed_features = model.forward_encoder(changed, mask)
        for actual, expected in zip(features, changed_features):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        target = model.patchify(image)
        prediction = target + (~mask).unsqueeze(-1) * 10
        self.assertEqual(model.forward_loss(image, prediction, mask).item(), 0.0)
        with self.assertRaisesRegex(ValueError, "At least one"):
            model.forward_loss(image, target, torch.zeros_like(mask))

    def test_backward_reaches_all_encoder_and_decoder_parameters(self):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        loss, prediction, mask = model(torch.randn(1, 11, 256, 256))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(prediction.shape), (1, 4096, 176))
        loss.backward()
        missing = [name for name, value in model.named_parameters() if value.grad is None]
        self.assertEqual(missing, [])
        for name, value in model.named_parameters():
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        self.assertGreater(model.backbone.embeddings.patch_embeddings.projection.weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)

    def test_export_strict_transfer_and_full_mask2former_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.pt"
            export_encoder(self.model.backbone, path)
            # Notebook historically uses config.image_size=224 but actual inputs=256.
            # No absolute embeddings: both configurations must have identical outputs.
            config = ai01_swin_config()
            config.image_size = 224
            target = Mask2FormerForUniversalSegmentation(Mask2FormerConfig(
                backbone_config=config.to_dict(), num_labels=2, num_queries=7,
            )).eval()
            report = load_ai01_encoder(target, path)
            self.assertEqual(report["loaded_tensors"], len(self.model.backbone.state_dict()))
            reference = load_exported_backbone(path).eval()
            images = torch.randn(1, 11, 256, 256)
            with torch.no_grad():
                actual = target.model.pixel_level_module.encoder(images).feature_maps
                expected = reference(images).feature_maps
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                outputs = target(pixel_values=images)
            self.assertEqual(tuple(outputs.class_queries_logits.shape), (1, 7, 3))
            self.assertEqual(tuple(outputs.masks_queries_logits.shape), (1, 7, 64, 64))
            # Reject incompatible metadata BEFORE copying any weights.
            projection = target.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.weight
            snapshot = projection.detach().clone()
            artifact = torch.load(path, weights_only=True)
            artifact["backbone_config"]["depths"] = [2, 2, 2, 2]
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "depths"):
                load_ai01_encoder(target, path)
            torch.testing.assert_close(projection, snapshot, rtol=0, atol=0)

    def test_reject_bad_export_shapes_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.pt"
            export_encoder(self.model.backbone, path)
            artifact = torch.load(path, weights_only=True)
            artifact["preprocessing"]["mean"] = 0.0
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "mean"):
                load_ai01_encoder(self.model.backbone, path)
            artifact["preprocessing"] = dict(DEFAULT_PREPROCESSING)
            original = artifact["encoder"]["embeddings.norm.weight"]
            artifact["encoder"]["embeddings.norm.weight"] = original[:1]
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "shapes"):
                load_ai01_encoder(self.model.backbone, path)
            artifact["encoder"]["embeddings.norm.weight"] = original
            del artifact["encoder"]["embeddings.norm.weight"]
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "missing"):
                load_ai01_encoder(self.model.backbone, path)


class NotebookTests(unittest.TestCase):
    def test_notebook_remains_valid_and_loads_ssl_after_projection(self):
        notebook_path = Path(__file__).resolve().parents[3] / "A1_Refractor.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for cell in code_cells:
            compile(cell, str(notebook_path), "exec")
        initialization = next(cell for cell in code_cells if "ssl_report = load_ai01_encoder" in cell)
        self.assertLess(initialization.index("patch_embeddings.projection ="),
                        initialization.index("ssl_report = load_ai01_encoder"))
        self.assertIn("configuration.backbone_config.num_channels = NUM_CHANNELS", initialization)


if __name__ == "__main__":
    unittest.main()

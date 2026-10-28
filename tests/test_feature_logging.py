"""Fixed samples/masks, numerical summaries, image cadence and RNG isolation."""

from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ai01_features import FeatureLogger, feature_statistics, select_feature_indices, feature_geometry, pca_rgb, feature_drift
from ai01_swin_mae import AI01SwinMAE
from train_ai01 import configure_logger, main, parser


class FixedDataset:
    augment = False
    files = [Path("well_a/0.npy"), Path("well_a/1.npy"), Path("well_b/0.npy"), Path("well_c/0.npy")]

    def __getitem__(self, index):
        return torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32).repeat(11, 1, 1) / 1024 + index


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, images):
        base = images.mean(dim=1, keepdim=True) * self.weight
        maps = tuple(F.adaptive_avg_pool2d(base, (size, size)).repeat(1, channels, 1, 1)
                     for size, channels in ((8, 4), (4, 8), (2, 16), (1, 32)))
        return SimpleNamespace(feature_maps=maps)


class TinyFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()

    def make_mask(self, batch_size, device, generator):
        return torch.rand(batch_size, 64, device=device, generator=generator) < 0.75

    def forward_encoder(self, images, mask):
        pixel_mask = mask.reshape(1, 1, 8, 8).repeat_interleave(4, -2).repeat_interleave(4, -1)
        return self.backbone(images.masked_fill(pixel_mask, 0)).feature_maps


class FeatureLoggingTests(unittest.TestCase):
    @contextmanager
    def make_logger(self, output):
        logger = configure_logger(output)
        try:
            yield logger
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def test_selection_prefers_distinct_wells_and_caps_count(self):
        self.assertEqual(select_feature_indices(FixedDataset.files, 3), [0, 2, 3])
        self.assertEqual(select_feature_indices(FixedDataset.files, 9), [0, 2, 3, 1])
        self.assertEqual(select_feature_indices(FixedDataset.files, 0), [])

    def test_statistics_and_population_spatial_standard_deviation(self):
        feature = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 5.0], [5.0, 5.0]]]])
        stats = feature_statistics(feature)
        self.assertEqual(stats["shape"], [1, 2, 2, 2])
        self.assertTrue(stats["finite"])
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertAlmostEqual(stats["mean"], feature.mean().item())
        self.assertAlmostEqual(stats["std"], feature.std(unbiased=False).item())
        self.assertAlmostEqual(stats["spatial_std"], 1.25 ** 0.5 / 2)

    def test_nonfinite_statistics_are_valid_strict_json(self):
        for value in (float("nan"), float("inf")):
            stats = feature_statistics(torch.tensor([[[[1.0, value]]]]))
            self.assertFalse(stats["finite"])
            self.assertEqual(stats["nonfinite_count"], 1)
            self.assertIsNone(stats["std"])
            json.dumps(stats, allow_nan=False)

    def test_fixed_masks_rng_weights_mode_and_jsonl_across_epochs(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), self.make_logger(temporary) as logger:
            output = Path(temporary)
            model = TinyFeatureModel().train()
            original_weight = model.backbone.weight.detach().clone()
            torch_rng = torch.get_rng_state().clone()
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            debug = FeatureLogger(model, FixedDataset(), output, logger, samples=3,
                                  image_freq=0, mode="both", seed=42)
            masks = [mask.clone() for _, mask, _ in debug.samples]
            debug.log_epoch(model, torch.device("cpu"), epoch=1, total_epochs=5)
            debug.log_epoch(model, torch.device("cpu"), epoch=2, total_epochs=5)
            self.assertTrue(model.training)
            self.assertIsNone(model.backbone.weight.grad)
            torch.testing.assert_close(model.backbone.weight, original_weight)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(random.getstate(), python_rng)
            current_numpy = np.random.get_state()
            self.assertEqual(numpy_rng[0], current_numpy[0])
            np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
            self.assertEqual(numpy_rng[2:], current_numpy[2:])
            for expected, (_, actual, _) in zip(masks, debug.samples):
                self.assertTrue(torch.equal(expected, actual))
            records = [json.loads(line) for line in (output / "feature_stats.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 12)
            self.assertEqual([row["sample_index"] for row in records[:6]], [0, 0, 2, 2, 3, 3])
            for first, second in zip(records[:6], records[6:]):
                self.assertEqual(first["mask_sha256"], second["mask_sha256"])
                self.assertEqual(first["stages"], second["stages"])
                self.assertEqual(set(first["stages"]), {"stage1", "stage2", "stage3", "stage4"})
                if first["mode"] == "unmasked":
                    self.assertIsNone(first["mask_seed"])
                    self.assertEqual(first["masked_fraction"], 0.0)
            self.assertEqual(list((output / "feature_debug").glob("*.png")), [])
            text = (output / "train.log").read_text(encoding="utf-8")
            self.assertIn("min=", text)
            self.assertIn("max=", text)
            self.assertIn("spatial_std=", text)
            metadata = json.loads((output / "feature_debug" / "samples.json").read_text())
            self.assertEqual([item["sample_index"] for item in metadata["samples"]], [0, 2, 3])
            reconstructed = FeatureLogger(model, FixedDataset(), output, logger, samples=3, image_freq=0)
            for expected, (_, actual, _) in zip(masks, reconstructed.samples):
                self.assertTrue(torch.equal(expected, actual))

    def test_png_cadence_and_real_headless_render(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), self.make_logger(temporary) as logger:
            output = Path(temporary)
            model = TinyFeatureModel().eval()
            torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
            debug = FeatureLogger(model, FixedDataset(), output, logger, samples=1, mode="masked")
            self.assertEqual([epoch for epoch in range(1, 13) if debug.save_images_at(epoch, 12)], [1, 5, 10, 12])
            debug.log_epoch(model, torch.device("cpu"), epoch=1, total_epochs=12)
            png = output / "feature_debug" / "epoch001_val0000_masked.png"
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertFalse(model.training)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(random.getstate(), python_rng)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
            with patch("ai01_features.save_feature_grid") as save:
                debug.log_epoch(model, torch.device("cpu"), epoch=2, total_epochs=12)
                save.assert_not_called()
                debug.log_epoch(model, torch.device("cpu"), epoch=12, total_epochs=12)
                save.assert_called_once()

    def test_mode_restored_if_debug_forward_fails(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), self.make_logger(temporary) as logger:
            output = Path(temporary)
            model = TinyFeatureModel().train()
            debug = FeatureLogger(model, FixedDataset(), output, logger, image_freq=0)
            with patch.object(model, "forward_encoder", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    debug.log_epoch(model, torch.device("cpu"), epoch=1, total_epochs=2)
            self.assertTrue(model.training)

    def test_224_unmasked_debug_does_not_mutate_swin_shift_window_state(self):
        torch.set_num_threads(2)
        model = AI01SwinMAE(image_size=224).eval()
        image = torch.randn(1, 11, 224, 224)

        class Dataset:
            augment = False
            files = [Path("well/frame.npy")]

            def __getitem__(self, index):
                return image[0]

        def runtime():
            return [(module.shift_size, module.window_size) for module in model.backbone.modules()
                    if hasattr(module, "shift_size")]

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), self.make_logger(temporary) as logger:
            debug = FeatureLogger(model, Dataset(), temporary, logger, samples=1, image_freq=0, mode="both")
            mask = debug.samples[0][1]
            before_runtime = runtime()
            with torch.inference_mode():
                expected = model.forward_encoder(image, mask)
                debug.log_epoch(model, torch.device("cpu"), epoch=1, total_epochs=2)
                actual = model.forward_encoder(image, mask)
            self.assertEqual(runtime(), before_runtime)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_defaults_and_invalid_cli_values(self):
        args = parser().parse_args(["--manifest", "unused.txt"])
        self.assertEqual((args.feature_samples, args.feature_image_freq, args.feature_mode, args.stage_log_freq), (3, 5, "both", 20))
        for flag in ("--feature_samples", "--feature_image_freq", "--stage_log_freq"):
            args = parser().parse_args(["--manifest", "unused.txt", flag, "-1"])
            with self.assertRaisesRegex(ValueError, "cannot be negative"):
                main(args)

    def test_geometry_detects_constant_tokens_and_rank_one_features(self):
        constant = torch.ones(1, 4, 4, 4)
        stats, basis = feature_geometry(constant)
        self.assertEqual(stats["effective_rank"], 0)
        self.assertEqual(stats["low_variance_channel_fraction"], 1)
        self.assertAlmostEqual(stats["mean_token_cosine"], 1)
        self.assertIsNone(stats["pca_top1_variance_ratio"])
        np.testing.assert_array_equal(pca_rgb(constant, basis), np.full((4, 4, 3), 0.5))
        values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        feature = values * torch.tensor([1.0, 2.0, 3.0, 0.0]).reshape(1, 4, 1, 1)
        stats, basis = feature_geometry(feature)
        self.assertAlmostEqual(stats["effective_rank"], 1)
        self.assertEqual(stats["max_centered_rank"], 4)
        self.assertAlmostEqual(stats["pca_top1_variance_ratio"], 1)
        self.assertAlmostEqual(stats["pca_top3_variance_ratio"], 1)
        self.assertEqual(stats["low_variance_channel_fraction"], 0.25)
        rgb = pca_rgb(feature, basis)
        self.assertEqual(rgb.shape, (4, 4, 3))
        self.assertTrue(np.isfinite(rgb).all())
        self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
        # Frozen basis/limits: changing the tensor changes RGB, not the basis.
        self.assertFalse(np.array_equal(rgb, pca_rgb(feature * 0.5, basis)))
        self.assertGreater(feature_drift(feature * 0.5, feature)["feature_rel_l2_from_baseline"], 0)

    def test_epoch_zero_reference_is_reused_on_resume_without_relogging(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), self.make_logger(temporary) as logger:
            model = TinyFeatureModel().train()
            debug = FeatureLogger(model, FixedDataset(), temporary, logger, samples=1, image_freq=0)
            debug.initialize_baseline(model, torch.device("cpu"), epoch=0, total_epochs=3, origin="random_init")
            original_basis = debug.references["0:unmasked:stage1"]["pca"]["components"].clone()
            with torch.no_grad():
                model.backbone.weight.mul_(2)
            debug.log_epoch(model, torch.device("cpu"), epoch=1, total_epochs=3)
            rows = [json.loads(line) for line in (Path(temporary) / "feature_stats.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["epoch"], 0)
            self.assertEqual(rows[0]["kind"], "baseline")
            self.assertEqual(rows[0]["stages"]["stage1"]["feature_rel_l2_from_baseline"], 0)
            self.assertGreater(rows[-1]["stages"]["stage1"]["feature_rel_l2_from_baseline"], 0)
            restored = FeatureLogger(model, FixedDataset(), temporary, logger, samples=1, image_freq=0)
            restored.initialize_baseline(model, torch.device("cpu"), epoch=1, total_epochs=3,
                                         origin="resume", resume_directory=temporary)
            self.assertEqual(restored.baseline_epoch, 0)
            self.assertEqual(restored.baseline_origin, "random_init")
            torch.testing.assert_close(restored.references["0:unmasked:stage1"]["pca"]["components"], original_basis)
            self.assertEqual(len((Path(temporary) / "feature_stats.jsonl").read_text().splitlines()), 4)
            with self.assertRaises(FileExistsError):
                restored.initialize_baseline(model, torch.device("cpu"), epoch=0, total_epochs=3, origin="random_init")

    def test_old_resume_without_reference_is_labeled_not_random_epoch_zero(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), self.make_logger(temporary) as logger:
            debug = FeatureLogger(TinyFeatureModel(), FixedDataset(), temporary, logger, samples=1, image_freq=0)
            debug.initialize_baseline(TinyFeatureModel(), torch.device("cpu"), epoch=7, total_epochs=10,
                                      origin="resume", resume_directory=temporary)
            rows = [json.loads(line) for line in (Path(temporary) / "feature_stats.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [7, 7])
            self.assertTrue(all(row["baseline_origin"] == "resume" and row["baseline_epoch"] == 7 for row in rows))


if __name__ == "__main__":
    unittest.main()

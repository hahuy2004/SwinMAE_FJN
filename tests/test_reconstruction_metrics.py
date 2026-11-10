"""Analytical metrics, masked-window correctness, aggregation and trainer parity."""

from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.ai01_metrics import ReconstructionMetricLogger, reconstruction_metrics
from v1.ai01_train_v1 import parser
from utils.ai01_train import validate


SPEC = {"num_channels": 2, "mean": 0.449, "std": 0.226, "image_size": 32}


def normalized(value):
    return (value - SPEC["mean"]) / SPEC["std"]


class TinyReconstructionModel(nn.Module):
    norm_pix_loss = False
    config = SimpleNamespace(patch_size=4)

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.1))

    def make_mask(self, batch_size, device, generator=None):
        return torch.rand(batch_size, 64, generator=generator, device=device) < 0.75

    def forward(self, images, mask=None):
        prediction = images + self.offset
        return self.offset.square(), prediction, mask

    def unpatchify(self, prediction):
        return prediction


class ReconstructionMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def images(self, batch=1):
        return normalized(torch.full((batch, 2, 32, 32), 0.25))

    def mask(self, batch=1):
        return torch.ones(batch, 64, dtype=torch.bool)

    def test_exact_match_and_known_constant_error(self):
        target = self.images()
        actual, counts = reconstruction_metrics(target, target.clone(), self.mask(), 4, SPEC)
        self.assertEqual(counts.item(), 26 * 26)
        torch.testing.assert_close(actual[0, :, 0], torch.full((2,), 120.0))
        torch.testing.assert_close(actual[0, :, 1], torch.ones(2))
        torch.testing.assert_close(actual[0, :, 2:], torch.zeros(2, 3))
        prediction = normalized(torch.full_like(target, 0.5))
        actual, _ = reconstruction_metrics(target, prediction, self.mask(), 4, SPEC)
        self.assertAlmostEqual(actual[0, 0, 0].item(), -20 * math.log10(0.25), places=4)
        self.assertAlmostEqual(actual[0, 0, 1].item(), (2 * 0.25 * 0.5 + 0.01 ** 2) / (0.25 ** 2 + 0.5 ** 2 + 0.01 ** 2), places=4)
        self.assertAlmostEqual(actual[0, 0, 2].item(), 63.75, places=4)
        self.assertAlmostEqual(actual[0, 0, 3].item(), 63.75, places=4)
        self.assertAlmostEqual(actual[0, 0, 4].item(), 0, places=5)

    def test_visible_errors_do_not_affect_any_masked_metric(self):
        target = self.images()
        mask = self.mask().reshape(1, 8, 8)
        mask[:, :, 4:] = False
        mask = mask.flatten(1)
        prediction = target.clone()
        prediction[..., 16:] = 1000
        actual, counts = reconstruction_metrics(target, prediction, mask, 4, SPEC)
        expected, _ = reconstruction_metrics(target, target, mask, 4, SPEC)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counts.item(), 26 * 10)

    def test_per_z_separation_and_gradient_error(self):
        target = self.images()
        prediction = target.clone()
        pattern = ((torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2).float() * 0.2
        prediction[:, 0] += pattern / SPEC["std"]
        actual, _ = reconstruction_metrics(target, prediction, self.mask(), 4, SPEC)
        self.assertAlmostEqual(actual[0, 0, 4].item(), 51, places=4)
        self.assertEqual(actual[0, 1, 0].item(), 120)
        self.assertEqual(actual[0, 1, 2].item(), 0)

    def test_ssim_matches_independent_population_window_reference(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(0.1, 0.9, (10, 10)).astype(np.float32)
        y = (0.8 * x + 0.05).astype(np.float32)
        reference = []
        for row in range(8):
            for col in range(8):
                a = x[row:row + 3, col:col + 3].astype(np.float64)
                b = y[row:row + 3, col:col + 3].astype(np.float64)
                covariance = ((a - a.mean()) * (b - b.mean())).mean()
                reference.append(((2 * a.mean() * b.mean() + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) /
                                 ((a.mean() ** 2 + b.mean() ** 2 + 0.01 ** 2) * (a.var() + b.var() + 0.03 ** 2)))
        spec = {**SPEC, "num_channels": 1}
        target, prediction = normalized(torch.from_numpy(x)[None, None]), normalized(torch.from_numpy(y)[None, None])
        actual, _ = reconstruction_metrics(target, prediction, torch.ones(1, 100, dtype=torch.bool), 1, spec, 3)
        self.assertAlmostEqual(actual[0, 0, 1].item(), float(np.mean(reference)), places=5)

    def test_no_eligible_windows_or_edges_are_explicitly_undefined(self):
        target = self.images()[..., :8, :8]
        checker = (torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2 == 0
        actual, counts = reconstruction_metrics(target, target, checker.flatten()[None], 1, SPEC, 3)
        self.assertEqual(counts.item(), 0)
        self.assertTrue(torch.isnan(actual[..., 1]).all())
        self.assertTrue(torch.isnan(actual[..., 4]).all())
        monitor = ReconstructionMetricLogger([Path("2/well_a/0.npy")], SPEC, ssim_window=3)
        monitor.update(target, target, checker.flatten()[None], 1, 0, 42)
        report = monitor.summary(1)
        self.assertEqual(report["frame_macro"]["masked_ssim"], {"mean": None, "std": None, "count": 0})
        self.assertEqual(report["ssim_eligible_windows_per_evaluation"]["evaluations_without_windows"], 1)
        json.dumps(report, allow_nan=False)

    def test_frame_well_group_z_and_seed_aggregation(self):
        files = [Path("2/well_a/0.npy"), Path("2/well_a/1.npy"), Path("3/well_b/0.npy")]
        monitor = ReconstructionMetricLogger(files, SPEC)
        target = self.images(3)
        errors = torch.tensor([0.1, 0.01, 0.001]).reshape(3, 1, 1, 1)
        prediction = target + errors / SPEC["std"]
        for seed in (42, 43):
            monitor.update(target, prediction, self.mask(3), 4, 0, seed)
        report = monitor.summary(100)
        self.assertEqual(report["num_frames"], 3)
        self.assertEqual(report["num_wells"], 2)
        self.assertEqual(report["num_evaluations"], 6)
        self.assertEqual(report["mask_seeds"], [42, 43])
        self.assertAlmostEqual(report["frame_macro"]["masked_psnr"]["mean"], 40, places=3)
        self.assertAlmostEqual(report["well_macro"]["masked_psnr"]["mean"], 45, places=3)
        self.assertAlmostEqual(report["well_macro"]["masked_psnr"]["std"], 15, places=3)
        self.assertEqual(report["well_macro"]["masked_psnr"]["count"], 2)
        self.assertEqual(report["per_group"]["2"]["num_frames"], 2)
        self.assertEqual(set(report["per_z"]), {"Z00", "Z01"})
        self.assertEqual(report["metric_definitions"]["masked_ssim"]["better"], "higher")

    def test_reports_have_directions_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = ReconstructionMetricLogger([Path("2/well_a/0.npy")], SPEC, output=directory)
            monitor.update(self.images(), self.images(), self.mask(), 4, 0, 42)
            report = monitor.write(1)
            root = Path(directory)
            latest = json.loads((root / "reconstruction_metrics_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(report, latest)
            self.assertEqual(len((root / "reconstruction_metrics.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            text = (root / "reconstruction_metrics.log").read_text(encoding="utf-8")
            self.assertIn("higher is better", text)
            self.assertIn("lower is better", text)
            self.assertIn("Càng thấp càng tốt", text)

    def test_invalid_inputs_incomplete_and_duplicate_evaluations(self):
        for window in (0, 2, 4):
            with self.assertRaisesRegex(ValueError, "odd"):
                reconstruction_metrics(self.images(), self.images(), self.mask(), 4, SPEC, window)
        with self.assertRaisesRegex(ValueError, "masked pixels"):
            reconstruction_metrics(self.images(), self.images(), torch.zeros_like(self.mask()), 4, SPEC)
        with self.assertRaises(FloatingPointError):
            reconstruction_metrics(self.images(), self.images() * float("nan"), self.mask(), 4, SPEC)
        monitor = ReconstructionMetricLogger([Path("2/a/0.npy"), Path("2/b/0.npy")], SPEC)
        monitor.update(self.images(), self.images(), self.mask(), 4, 0, 42)
        with self.assertRaisesRegex(ValueError, "every validation"):
            monitor.summary(1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            monitor.update(self.images(), self.images(), self.mask(), 4, 0, 42)

    def test_validation_loss_masks_weights_and_rng_unchanged_by_metrics(self):
        model = TinyReconstructionModel()
        images = self.images(3)
        loader = DataLoader(list(images), batch_size=2, generator=torch.Generator().manual_seed(0))
        initial = model.offset.detach().clone()
        with redirect_stdout(io.StringIO()):
            loss_without = validate(model, loader, torch.device("cpu"), 42, False)
            monitor = ReconstructionMetricLogger([Path(f"2/well_{i}/0.npy") for i in range(3)], SPEC)
            loss_with = validate(model, loader, torch.device("cpu"), 42, False, reconstruction_monitor=monitor)
        self.assertEqual(loss_without, loss_with)
        torch.testing.assert_close(model.offset, initial)
        self.assertIsNone(model.offset.grad)
        report = monitor.summary(1)
        self.assertEqual(report["num_frames"], 3)
        self.assertEqual(report["well_macro"]["masked_mae"]["count"], 3)
        state = torch.get_rng_state().clone()
        reconstruction_metrics(self.images(), self.images(), self.mask(), 4, SPEC)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_defaults_and_normalized_target_guard(self):
        args = parser().parse_args(["--manifest", "unused"])
        self.assertFalse(args.no_reconstruction_metrics)
        self.assertEqual(args.ssim_window, 7)
        model = TinyReconstructionModel()
        model.norm_pix_loss = True
        monitor = ReconstructionMetricLogger([Path("2/a/0.npy")], SPEC)
        with self.assertRaisesRegex(ValueError, "norm_pix_loss"):
            validate(model, DataLoader(list(self.images())), torch.device("cpu"), 42, False,
                     reconstruction_monitor=monitor)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_cpu_parity(self):
        target = self.images()
        prediction = target + 0.05
        expected, _ = reconstruction_metrics(target, prediction, self.mask(), 4, SPEC)
        actual, _ = reconstruction_metrics(target.cuda(), prediction.cuda(), self.mask().cuda(), 4, SPEC)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()

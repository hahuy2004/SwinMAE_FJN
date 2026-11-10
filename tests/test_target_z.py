"""Only Z05-Z07 supervise reconstruction; input/export retain all eleven Z."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils.ai01_data import DEFAULT_PREPROCESSING
from utils.ai01_metrics import ReconstructionMetricLogger, reconstruction_metrics
from v1.ai01_model_v1 import AI01SwinMAE, checkpoint_model_config
from utils.ai01_targets import validate_target_z
from v1.ai01_evaluate_v1 import main as evaluate_main, parser as evaluation_parser
from v1.ai01_train_v1 import parser


class TargetZTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = AI01SwinMAE(image_size=32)

    def setUp(self):
        self.model.norm_pix_loss = False
        self.model.zero_grad(set_to_none=True)

    def test_default_and_cli_targets_are_zero_based_5_6_7(self):
        self.assertEqual(self.model.target_z, (5, 6, 7))
        self.assertEqual(self.model.config.num_channels, 11)
        self.assertEqual(self.model.decoder.prediction.out_features, 16 * 11)
        self.assertEqual(parser().parse_args(["--manifest", "unused"]).target_z, [5, 6, 7])
        self.assertEqual(self.model.training_config()["target_z"], [5, 6, 7])

    def test_unselected_prediction_errors_are_ignored_and_selected_errors_are_averaged(self):
        images = torch.randn(2, 11, 32, 32)
        target = self.model.patchify(images)
        prediction = target.clone().reshape(2, 64, 16, 11)
        unused = [z for z in range(11) if z not in self.model.target_z]
        prediction[..., unused] += 100
        mask = torch.ones(2, 64, dtype=torch.bool)
        self.assertEqual(self.model.forward_loss(images, prediction.flatten(-2), mask).item(), 0)
        for z, error in ((5, 1), (6, 2), (7, 3)):
            prediction[..., z] += error
        self.assertAlmostEqual(self.model.forward_loss(images, prediction.flatten(-2), mask).item(), 14 / 3, places=5)
        mask[:, :32] = False
        prediction[:, :32, :, 5:8] += 1000
        self.assertAlmostEqual(self.model.forward_loss(images, prediction.flatten(-2), mask).item(), 14 / 3, places=5)

    def test_output_gradients_exist_only_for_selected_z_at_masked_patches(self):
        images = torch.randn(1, 11, 32, 32)
        prediction = torch.randn_like(self.model.patchify(images), requires_grad=True)
        mask = torch.ones(1, 64, dtype=torch.bool)
        mask[:, :32] = False
        self.model.forward_loss(images, prediction, mask).backward()
        gradient = prediction.grad.reshape(1, 64, 16, 11)
        unused = [z for z in range(11) if z not in self.model.target_z]
        self.assertEqual(gradient[..., unused].abs().sum().item(), 0)
        self.assertEqual(gradient[:, :32].abs().sum().item(), 0)
        self.assertGreater(gradient[..., 5:8].abs().sum().item(), 0)

    def test_unselected_target_z_do_not_enter_optional_patch_normalization(self):
        self.model.norm_pix_loss = True
        images = torch.randn(1, 11, 32, 32)
        changed = images.clone()
        unused = [z for z in range(11) if z not in self.model.target_z]
        changed[:, unused] += 1000
        prediction = torch.randn_like(self.model.patchify(images))
        mask = torch.ones(1, 64, dtype=torch.bool)
        torch.testing.assert_close(self.model.forward_loss(images, prediction, mask),
                                   self.model.forward_loss(changed, prediction, mask), rtol=0, atol=0)

    def test_all_eleven_input_channels_receive_gradients_and_inference_shapes_stay_unchanged(self):
        self.model.train()
        images = torch.randn(2, 11, 32, 32)
        loss, prediction, mask = self.model(images)
        self.assertEqual(tuple(prediction.shape), (2, 64, 176))
        self.assertEqual(tuple(self.model.unpatchify(prediction).shape), (2, 11, 32, 32))
        loss.backward()
        gradient = self.model.backbone.embeddings.patch_embeddings.projection.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum((0, 2, 3)) > 0).all())
        # The same spatial mask is applied before encoder stages, across all Z.
        self.assertEqual(tuple(mask.shape), (2, 64))

    def test_legacy_checkpoint_semantics_and_architecture_compatibility(self):
        legacy = checkpoint_model_config({"mask_ratio": 0.75})
        self.assertEqual(legacy["target_z"], list(range(11)))
        self.assertEqual(legacy["image_size"], 256)
        saved = checkpoint_model_config(self.model.training_config())
        self.assertEqual(saved["target_z"], [5, 6, 7])
        restored = AI01SwinMAE(**saved)
        restored.load_state_dict(self.model.state_dict(), strict=True)
        all_z = AI01SwinMAE(image_size=32, target_z=list(range(11)))
        all_z.load_state_dict(self.model.state_dict(), strict=True)
        images = torch.randn(1, 11, 32, 32)
        target = all_z.patchify(images)
        prediction = target + torch.randn_like(target)
        mask = all_z.make_mask(1, "cpu")
        original_mse = ((prediction - target).square().mean(-1) * mask).sum() / mask.sum()
        torch.testing.assert_close(all_z.forward_loss(images, prediction, mask), original_mse, rtol=0, atol=0)

    def test_invalid_target_indices(self):
        for values in ([], [5, 5, 6], [-1, 6, 7], [5, 6, 11], [True], [5.0], None):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_target_z(values, 11)
        self.assertEqual(validate_target_z([7, 5, 6], 11), (5, 6, 7))

    def test_metrics_ignore_unsupervised_z_and_keep_correct_z_labels(self):
        spec = {**DEFAULT_PREPROCESSING, "image_size": 32}
        images = torch.zeros(1, 11, 32, 32)
        predictions = images.clone()
        unused = [z for z in range(11) if z not in self.model.target_z]
        predictions[:, unused] = 100
        mask = torch.ones(1, 64, dtype=torch.bool)
        actual, _ = reconstruction_metrics(images, predictions, mask, 4, spec, target_z=self.model.target_z)
        self.assertEqual(tuple(actual.shape), (1, 3, 5))
        self.assertTrue((actual[..., 0] == 120).all())
        self.assertTrue((actual[..., 2:] == 0).all())
        monitor = ReconstructionMetricLogger([Path("2/well/frame.npy")], spec, target_z=self.model.target_z)
        monitor.update(images, predictions, mask, 4, 0, 42)
        report = monitor.summary(1)
        self.assertEqual(report["input_num_z"], 11)
        self.assertEqual(report["num_z"], 3)
        self.assertEqual(report["target_z"], [5, 6, 7])
        self.assertEqual(set(report["per_z"]), {"Z05", "Z06", "Z07"})
        self.assertEqual(report["well_macro"]["masked_mae"]["mean"], 0)

    def test_evaluator_rejects_untrained_target_z_before_running_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"format": "ai01-swin-mae-training-v1", "model_config": self.model.training_config(),
                        "preprocessing": DEFAULT_PREPROCESSING}, path)
            args = evaluation_parser().parse_args(["--checkpoint", str(path), "--val_manifest", "unused",
                                                   "--output_dir", str(Path(directory) / "eval"),
                                                   "--target_z", "0", "--device", "cpu"])
            with self.assertRaisesRegex(ValueError, "not supervised"):
                evaluate_main(args)


if __name__ == "__main__":
    unittest.main()

"""Offline V2 architecture/objective/transfer and real train-resume-eval tests."""

from contextlib import contextmanager, redirect_stdout
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import SwinBackbone, Swinv2Backbone
from transformers.models.swinv2.modeling_swinv2 import Swinv2Layer

from v1.ai01_model_v1 import ai01_swin_config
from v1.ai01_transfer_v1 import export_encoder as export_v1, load_ai01_encoder
from v2.ai01_learning_v2 import AI01V2StageLearningMonitor, optimizer_parameter_groups
from v2.ai01_model_v2 import AI01V2SwinMAE, TRAINING_FORMAT, ai01_v2_swin_config
from v2.ai01_transfer_v2 import (
    build_ai01_v2_mask2former, export_encoder, load_ai01_v2_encoder,
    load_exported_backbone,
)
from v2 import ai01_evaluate_v2 as ai01_v2_evaluate
from v2 import ai01_train_v2 as ai01_v2_train
from utils import ai01_train as train_ai01


def small_config(image_size=64, window_size=2, drop_path_rate=0.0):
    config = ai01_v2_swin_config(image_size, window_size, drop_path_rate)
    config.embed_dim = 12
    config.depths = [2, 2, 2, 2]
    config.num_heads = [1, 2, 4, 8]
    return config


@contextmanager
def close_training_logger():
    try:
        yield
    finally:
        logger = logging.getLogger("ai01_ssl")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_default_tiny_shapes_mask_and_all_parameter_backward(self):
        model = AI01V2SwinMAE(decoder_depth=1)
        self.assertIsInstance(model.backbone, Swinv2Backbone)
        self.assertEqual(model.config.depths, [2, 2, 6, 2])
        self.assertEqual(model.config.window_size, 8)
        self.assertFalse(hasattr(model.backbone, "hidden_states_norms"))
        merging = model.backbone.encoder.layers[0].downsample
        self.assertEqual(merging.norm.normalized_shape, (192,))
        images = torch.randn(1, 11, 256, 256)
        torch.testing.assert_close(model.unpatchify(model.patchify(images)), images, rtol=0, atol=0)
        mask = model.make_mask(1, "cpu", torch.Generator().manual_seed(5))
        self.assertEqual(mask.sum().item(), 3072)
        grouped = mask.reshape(1, 16, 4, 16, 4).permute(0, 1, 3, 2, 4)
        self.assertTrue(torch.equal(grouped, grouped[..., :1, :1].expand_as(grouped)))
        model.eval()
        with torch.no_grad():
            features = model.forward_encoder(images, torch.zeros_like(mask))
            reference = model.backbone(images).feature_maps
        self.assertEqual([tuple(x.shape) for x in features], [
            (1, 96, 64, 64), (1, 192, 32, 32), (1, 384, 16, 16), (1, 768, 8, 8),
        ])
        for left, right in zip(features, reference):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        model.train()
        loss, prediction, returned_mask = model(images, mask)
        self.assertEqual(tuple(prediction.shape), (1, 4096, 176))
        self.assertTrue(torch.equal(mask, returned_mask))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        # Official zero-initialized residual norms deliberately gate attention
        # parameter gradients on step 1. The residual norms themselves learn.
        norm = model.backbone.encoder.layers[0].blocks[0].layernorm_before
        self.assertGreater(norm.weight.grad.abs().sum().item(), 0)

    def test_official_cosine_attention_log_cpb_and_residual_post_norm(self):
        model = AI01V2SwinMAE(encoder_config=small_config(), decoder_depth=1)
        block = model.backbone.encoder.layers[0].blocks[0]
        attention = block.attention.self
        self.assertIsNone(attention.key.bias)
        self.assertTrue(torch.equal(block.layernorm_before.weight, torch.zeros(12)))
        tokens = torch.randn(2, 4, 12)
        with torch.no_grad():
            actual = attention(tokens)[0]
            q = attention.transpose_for_scores(attention.query(tokens))
            k = attention.transpose_for_scores(attention.key(tokens))
            v = attention.transpose_for_scores(attention.value(tokens))
            logits = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(k, dim=-1).transpose(-2, -1)
            logits *= attention.logit_scale.clamp(max=math.log(100)).exp()
            table = attention.continuous_position_bias_mlp(attention.relative_coords_table).view(-1, 1)
            bias = 16 * table[attention.relative_position_index.flatten()].view(4, 4, 1).permute(2, 0, 1).sigmoid()
            expected = ((logits + bias).softmax(-1) @ v).permute(0, 2, 1, 3).reshape(2, 4, 12)
            torch.testing.assert_close(actual, expected)
            coords = attention.relative_coords_table
            self.assertTrue(torch.isfinite(coords).all())
            self.assertLessEqual(coords.abs().max().item(), math.log2(9) / math.log2(8) + 1e-6)
            # Nonzero norms ensure the block comparison tests actual attention,
            # not only its zero-init identity behavior.
            block.layernorm_before.weight.fill_(1)
            block.layernorm_after.weight.fill_(1)
            x = torch.randn(1, 16, 12)
            spatial = x.reshape(1, 4, 4, 12)
            windows = spatial.reshape(1, 2, 2, 2, 2, 12).permute(0, 1, 3, 2, 4, 5).reshape(4, 4, 12)
            attended = block.attention(windows)[0].reshape(1, 2, 2, 2, 2, 12)
            attended = attended.permute(0, 1, 3, 2, 4, 5).reshape(1, 16, 12)
            residual = x + block.layernorm_before(attended)
            expected = residual + block.layernorm_after(block.output(block.intermediate(residual)))
            torch.testing.assert_close(block(x, (4, 4))[0], expected)

    def test_masked_pixels_do_not_leak_and_only_target_z_contribute(self):
        model = AI01V2SwinMAE(encoder_config=small_config(), decoder_depth=1).eval()
        for module in model.modules():
            if isinstance(module, Swinv2Layer):
                module.layernorm_before.weight.data.fill_(1)
                module.layernorm_after.weight.data.fill_(1)
        image = torch.randn(2, 11, 64, 64)
        mask = model.make_mask(2, "cpu")
        image_mask = mask.reshape(2, 1, 16, 16).repeat_interleave(4, -2).repeat_interleave(4, -1)
        changed = torch.where(image_mask, torch.full_like(image, 100), image)
        with torch.no_grad():
            for left, right in zip(model.forward_encoder(image, mask), model.forward_encoder(changed, mask)):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        target = model.patchify(image)
        prediction = target.clone().reshape(2, 256, 16, 11)
        prediction[..., [0, 1, 2, 3, 4, 8, 9, 10]] += 100
        prediction = prediction.reshape_as(target) + (~mask).unsqueeze(-1) * 30
        self.assertEqual(model.forward_loss(image, prediction, mask).item(), 0)
        with self.assertRaisesRegex(ValueError, "At least one"):
            model.forward_loss(image, target, torch.zeros_like(mask))
        model.norm_pix_loss = True
        loss = model.forward_loss(image, prediction, mask)
        changed_targets = image.clone()
        changed_targets[:, [0, 1, 2, 3, 4, 8, 9, 10]] += 1000
        torch.testing.assert_close(loss, model.forward_loss(changed_targets, prediction, mask), rtol=0, atol=0)

    def test_224_padding_is_stable_and_unmasked_path_matches(self):
        model = AI01V2SwinMAE(encoder_config=small_config(224, 8), decoder_depth=1).eval()
        image = torch.randn(1, 11, 224, 224)
        mask = torch.zeros(1, 56 * 56, dtype=torch.bool)
        with torch.no_grad():
            for left, right in zip(model.forward_encoder(image, mask), model.backbone(image).feature_maps):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            loss, prediction, _ = model(image)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(prediction.shape), (1, 3136, 176))
        last = model.backbone.encoder.layers[-1].blocks[-1]
        self.assertEqual((last.window_size, last.shift_size), (7, 0))

    def test_attention_parameters_learn_after_zero_init_and_decay_exemptions(self):
        model = AI01V2SwinMAE(encoder_config=small_config(), decoder_depth=1)
        groups = optimizer_parameter_groups(model, 0.05)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        exempt = {id(p) for p in groups[1]["params"]}
        for name, parameter in model.named_parameters():
            if "logit_scale" in name or "continuous_position_bias_mlp" in name:
                self.assertIn(id(parameter), exempt)
        optimizer = torch.optim.AdamW(groups, lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(1, 11, 64, 64))[0].backward()
            optimizer.step()
        attention = model.backbone.encoder.layers[0].blocks[0].attention.self
        self.assertGreater(attention.query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(attention.continuous_position_bias_mlp[0].weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as temporary:
            monitor = AI01V2StageLearningMonitor(model.backbone, temporary, logging.getLogger("test"))
            self.assertEqual({id(p) for group in monitor.groups.values() for p in group},
                             {id(p) for p in model.backbone.parameters()})

    def test_strict_export_and_full_mask2former_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.pt"
            backbone = Swinv2Backbone(small_config(256, 8)).eval()
            export_encoder(backbone, path)
            reference = load_exported_backbone(path).eval()
            downstream = build_ai01_v2_mask2former(path, num_labels=2, num_queries=7).eval()
            images = torch.randn(1, 11, 256, 256)
            with torch.no_grad():
                for expected, actual in zip(reference(images).feature_maps,
                                            downstream.model.pixel_level_module.encoder(images).feature_maps):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                outputs = downstream(pixel_values=images)
            self.assertEqual(tuple(outputs.class_queries_logits.shape), (1, 7, 3))
            self.assertEqual(tuple(outputs.masks_queries_logits.shape), (1, 7, 64, 64))
            self.assertTrue(torch.isfinite(outputs.masks_queries_logits).all())
            projection = backbone.embeddings.patch_embeddings.projection.weight
            before = projection.detach().clone()
            artifact = torch.load(path, weights_only=True)
            artifact["backbone_config"]["window_size"] = 4
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "window_size"):
                load_ai01_v2_encoder(backbone, path)
            torch.testing.assert_close(projection, before, rtol=0, atol=0)
            export_encoder(backbone, path)
            artifact = torch.load(path, weights_only=True)
            artifact["encoder"]["embeddings.norm.weight"] = torch.zeros(1)
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "shapes"):
                load_ai01_v2_encoder(backbone, path)
            torch.testing.assert_close(projection, before, rtol=0, atol=0)
            v1_config = ai01_swin_config()
            v1_config.embed_dim, v1_config.depths, v1_config.num_heads = 12, [1, 1, 1, 1], [1, 2, 4, 8]
            v1 = SwinBackbone(v1_config)
            with self.assertRaisesRegex(TypeError, "Swinv2Backbone"):
                load_ai01_v2_encoder(v1, path)
            export_v1(v1, path)
            with self.assertRaisesRegex(ValueError, "V2 encoder export"):
                load_ai01_v2_encoder(backbone, path)
            export_encoder(backbone, path)
            with self.assertRaises(ValueError):
                load_ai01_encoder(v1, path)

    def test_supervised_mask2former_loss_reaches_v2_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.pt"
            export_encoder(Swinv2Backbone(small_config(256, 8)), path)
            model = build_ai01_v2_mask2former(
                path, num_labels=2, num_queries=7, train_num_points=64,
            ).train()
            masks = torch.zeros(2, 256, 256)
            masks[0, 30:80, 40:100] = 1
            masks[1, 140:190, 150:210] = 1
            outputs = model(
                pixel_values=torch.randn(1, 11, 256, 256),
                mask_labels=[masks], class_labels=[torch.tensor([0, 1])],
            )
            self.assertTrue(torch.isfinite(outputs.loss))
            outputs.loss.backward()
            encoder = model.model.pixel_level_module.encoder
            projection = encoder.embeddings.patch_embeddings.projection.weight
            self.assertIsNotNone(projection.grad)
            self.assertTrue(torch.isfinite(projection.grad).all())
            self.assertGreater(projection.grad.abs().sum().item(), 0)
            for name, parameter in encoder.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_v2_config_round_trip_and_invalid_inputs(self):
        model = AI01V2SwinMAE(encoder_config=small_config(), decoder_depth=1).eval()
        restored = AI01V2SwinMAE(**model.training_config()).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        image = torch.randn(1, 11, 64, 64)
        mask = model.make_mask(1, "cpu")
        with torch.no_grad():
            torch.testing.assert_close(model(image, mask)[1], restored(image, mask)[1], rtol=0, atol=0)
        for settings in ({"mask_ratio": 1}, {"mask_group_size": 3}, {"decoder_depth": 0},
                         {"target_z": [5, 5]}, {"target_z": [11]}):
            with self.assertRaises(ValueError):
                AI01V2SwinMAE(encoder_config=small_config(), **settings)
        with self.assertRaises(TypeError):
            AI01V2SwinMAE(encoder_config=ai01_swin_config())

    def test_cli_training_resume_evaluation_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary, close_training_logger(), redirect_stdout(io.StringIO()):
            root = Path(temporary)
            for well in ("train_well", "val_well"):
                (root / well).mkdir()
                np.save(root / well / "frame.npy", np.random.default_rng(42).integers(
                    0, 256, (11, 256, 256), dtype=np.uint8,
                ))
            for name, well in (("train", "train_well"), ("val", "val_well")):
                (root / f"{name}.txt").write_text(f"{well}/frame.npy\n", encoding="utf-8")
            output = root / "run"
            command = ["--manifest", str(root / "train.txt"), "--val_manifest", str(root / "val.txt"),
                       "--output_dir", str(output), "--epochs", "2", "--batch_size", "1", "--max_steps", "1",
                       "--device", "cpu", "--feature_samples", "1", "--feature_image_freq", "0",
                       "--decoder_depth", "1", "--gradient_checkpointing", "--no_augment"]
            original_epoch = train_ai01.train_epoch

            def stop_after_first(*args, **kwargs):
                if args[6] == 1:
                    raise InterruptedError("simulated epoch-boundary stop")
                return original_epoch(*args, **kwargs)

            with patch("v2.ai01_train_v2.ai01_v2_swin_config", side_effect=small_config):
                with patch("utils.ai01_train.train_epoch", side_effect=stop_after_first):
                    with self.assertRaises(InterruptedError):
                        ai01_v2_train.main(ai01_v2_train.parser().parse_args(command))
                state = torch.load(output / "checkpoint-last.pt", weights_only=True)
                self.assertEqual(state["format"], TRAINING_FORMAT)
                self.assertEqual(state["epoch"], 1)
                command += ["--resume", str(output / "checkpoint-last.pt")]
                ai01_v2_train.main(ai01_v2_train.parser().parse_args(command))
                # A completed resume still validates/restores every config/state.
                ai01_v2_train.main(ai01_v2_train.parser().parse_args(command))
            state = torch.load(output / "checkpoint-last.pt", weights_only=True)
            self.assertEqual(state["epoch"], 2)
            self.assertTrue((output / "encoder-best.pt").is_file())
            self.assertTrue((output / "encoder-last.pt").is_file())
            self.assertTrue((output / "feature_stats.jsonl").is_file())
            self.assertTrue((output / "stage_learning.jsonl").is_file())
            report = ai01_v2_evaluate.main(ai01_v2_evaluate.parser().parse_args([
                "--checkpoint", str(output / "checkpoint-last.pt"), "--val_manifest", str(root / "val.txt"),
                "--output_dir", str(root / "eval"), "--device", "cpu", "--batch_size", "1",
            ]))
            self.assertEqual(report["target_z"], [5, 6, 7])
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["backbone_config"]["model_type"], "swinv2")


if __name__ == "__main__":
    unittest.main()

"""Stage gradient norms are unscaled/pre-clip; weight deltas reflect real steps."""

from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from ai01_learning import StageLearningMonitor, stage_parameters
from ai01_swin_mae import AI01SwinMAE
from train_ai01 import configure_logger, parser, train_epoch


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Linear(1, 1, bias=False)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(nn.Linear(1, 1, bias=False) for _ in range(4))
        self.hidden_states_norms = nn.ModuleDict({f"stage{i}": nn.Identity() for i in range(1, 5)})
        for parameter in self.parameters():
            nn.init.constant_(parameter, 0.8)

    def forward(self, images):
        x = self.embeddings(images)
        for layer in self.encoder.layers:
            x = layer(x)
        return x


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()

    def forward(self, images):
        return self.backbone(images).square().mean(), None, None


class StageLearningTests(unittest.TestCase):
    @contextmanager
    def logger(self, directory):
        logger = configure_logger(directory)
        try:
            yield logger
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def args(self):
        return parser().parse_args(["--manifest", "unused", "--epochs", "2", "--warmup_epochs", "0",
                                    "--accum_iter", "2", "--lr", "0.001", "--clip_grad", "0.0001"])

    def test_real_swin_groups_cover_all_parameters_exactly_once(self):
        model = AI01SwinMAE()
        grouped = [id(parameter) for group in stage_parameters(model.backbone).values() for parameter in group]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(parameter) for parameter in model.backbone.parameters()})

    def test_gradient_cadence_frozen_stage_and_actual_weight_changes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), self.logger(directory) as logger:
            model = TinyModel()
            model.backbone.encoder.layers[3].weight.requires_grad_(False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            monitor = StageLearningMonitor(model.backbone, directory, logger, log_freq=2)
            train_epoch(model, DataLoader(torch.ones(5, 1)), optimizer, scaler, torch.device("cpu"),
                        self.args(), epoch=0, use_amp=False, learning_monitor=monitor)
            summary = monitor.end_epoch()
            rows = [json.loads(line) for line in (Path(directory) / "stage_learning.jsonl").read_text().splitlines()]
            self.assertEqual([row["update"] for row in rows[:-1]], [1, 2, 3])
            self.assertGreater(rows[0]["stages"]["stage1"]["grad_norm"], self.args().clip_grad)
            self.assertEqual(summary["successful_updates"], 3)
            for name in ("stage1", "stage2", "stage3"):
                self.assertTrue(summary["stages"][name]["changed"])
                self.assertGreater(summary["stages"][name]["weight_rel_change"], 0)
            self.assertFalse(summary["stages"]["stage4"]["changed"])
            self.assertIsNone(rows[0]["stages"]["stage4"]["grad_norm"])
            self.assertEqual(rows[0]["stages"]["stage4"]["parameters_without_grad"], 1)

    def test_amp_skip_is_counted_and_does_not_report_weight_changes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), self.logger(directory) as logger:
            model = TinyModel()
            model.backbone.embeddings.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            scaler = torch.amp.GradScaler("cpu", init_scale=65536)
            monitor = StageLearningMonitor(model.backbone, directory, logger)
            train_epoch(model, DataLoader(torch.ones(2, 1)), optimizer, scaler, torch.device("cpu"),
                        self.args(), epoch=0, use_amp=False, learning_monitor=monitor)
            summary = monitor.end_epoch()
            self.assertEqual(summary["attempted_updates"], 1)
            self.assertEqual(summary["skipped_updates"], 1)
            self.assertEqual(summary["successful_updates"], 0)
            self.assertTrue(all(not item["changed"] for item in summary["stages"].values()))
            first = json.loads((Path(directory) / "stage_learning.jsonl").read_text().splitlines()[0])
            self.assertTrue(first["optimizer_step_skipped"])
            self.assertIsNone(first["stages"]["stage1"]["grad_norm"])
            self.assertFalse(first["stages"]["stage1"]["grad_finite"])
            self.assertEqual((first["scale_before"], first["scale_after"]), (65536, 32768))


if __name__ == "__main__":
    unittest.main()

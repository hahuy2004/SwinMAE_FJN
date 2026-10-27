"""Fast CPU tests for CLI progress logging and gradient accumulation."""

from contextlib import redirect_stdout
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

from train_ai01 import configure_logger, log_progress, main, parser, train_epoch, validate


class TinyMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, images, mask=None):
        return (self.weight * images).square().mean(), None, mask

    def make_mask(self, batch_size, device, generator=None):
        return torch.zeros(batch_size, 1, dtype=torch.bool, device=device)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.001)
        self.updates = 0

    def step(self, closure=None):
        self.updates += 1
        return super().step(closure)


class LoggingTests(unittest.TestCase):
    def test_train_logs_first_interval_and_last_batch_and_keeps_partial_accumulation(self):
        args = parser().parse_args([
            "--manifest", "unused.txt", "--epochs", "2", "--warmup_epochs", "0",
            "--accum_iter", "2", "--log_freq", "2", "--lr", "0.001",
        ])
        model = TinyMAE()
        optimizer = CountingSGD(model.parameters())
        loader = DataLoader(torch.ones(5, 1), batch_size=1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "steps.jsonl"
            with redirect_stdout(io.StringIO()):
                loss = train_epoch(model, loader, optimizer, scaler, torch.device("cpu"), args,
                                   epoch=1, use_amp=False, step_log=output)
            records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["step"] for row in records], [1, 2, 4, 5])
        self.assertEqual([row["global_step"] for row in records], [6, 7, 9, 10])
        self.assertTrue(all(row["phase"] == "train" and row["epoch"] == 2 for row in records))
        self.assertEqual(records[-1]["samples_seen"], 5)
        self.assertAlmostEqual(records[-1]["loss_avg"], loss)
        self.assertEqual(records[-1]["eta_seconds"], 0.0)
        self.assertEqual(optimizer.updates, 3)
        self.assertLess(model.weight.item(), 0.5)

    def test_validation_progress_does_not_update_weights(self):
        model = TinyMAE()
        original = model.weight.detach().clone()
        loader = DataLoader(torch.ones(5, 1), batch_size=1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "steps.jsonl"
            with redirect_stdout(io.StringIO()):
                loss = validate(model, loader, torch.device("cpu"), 42, False,
                                epoch=2, log_freq=2, step_log=output)
            records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["step"] for row in records], [1, 2, 4, 5])
        self.assertTrue(all(row["phase"] == "validation" and row["epoch"] == 3 for row in records))
        self.assertAlmostEqual(loss, 0.25)
        torch.testing.assert_close(model.weight, original)
        self.assertIsNone(model.weight.grad)

    def test_console_log_and_json_metrics_are_both_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with redirect_stdout(io.StringIO()) as console:
                logger = configure_logger(output)
                try:
                    logger.info("Starting test run")
                    log_progress({
                        "phase": "train", "epoch": 1, "step": 10, "total_steps": 20,
                        "loss": 0.3, "loss_avg": 0.4, "lr": 1e-4,
                        "elapsed_seconds": 5.0, "eta_seconds": 5.0,
                    }, output / "steps.jsonl", logger)
                finally:
                    for handler in list(logger.handlers):
                        logger.removeHandler(handler)
                        handler.close()
            text = (output / "train.log").read_text(encoding="utf-8")
            self.assertIn("Starting test run", text)
            self.assertIn("step=10/20", text)
            self.assertIn("lr=1.000e-04", console.getvalue())
            self.assertEqual(json.loads((output / "steps.jsonl").read_text())["step"], 10)

    def test_log_frequency_must_be_positive(self):
        args = parser().parse_args(["--manifest", "unused.txt", "--log_freq", "0"])
        with self.assertRaisesRegex(ValueError, "log_freq must be positive"):
            main(args)


if __name__ == "__main__":
    unittest.main()

"""Regression tests: GradScaler must handle non-finite scaled gradients."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_ai01 import parser, train_epoch


class TinyMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, images):
        return (self.weight * images).square().mean(), None, None


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.001)
        self.weights_before_step = []

    def step(self, closure=None):
        self.weights_before_step.append(self.param_groups[0]["params"][0].detach().clone())
        return super().step(closure)


class HalfPrecisionMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.projection.weight, 0.5)

    def forward(self, images):
        prediction = self.projection(images)
        # Real fp16 backward overflows at scale=65536, while loss stays finite.
        return prediction.float().square().mean(), None, None


class AMPOverflowTests(unittest.TestCase):
    def args(self, accumulation=1):
        return parser().parse_args([
            "--manifest", "unused.txt", "--epochs", "2", "--warmup_epochs", "0",
            "--accum_iter", str(accumulation), "--lr", "0.001",
        ])

    def check_scaler_recovery(self, device):
        # Exercise the real GradScaler, not a mock; CPU also works in CPU-only CI.
        for bad_gradient in (float("inf"), float("nan")):
            with self.subTest(device=device, bad_gradient=bad_gradient):
                model = TinyMAE().to(device)
                original = model.weight.detach().clone()
                calls = 0

                def inject_first_backward(gradient):
                    nonlocal calls
                    calls += 1
                    return torch.full_like(gradient, bad_gradient) if calls == 1 else gradient

                model.weight.register_hook(inject_first_backward)
                optimizer = CountingSGD(model.parameters())
                scaler = torch.amp.GradScaler(device.type, init_scale=65536.0)
                loader = DataLoader(torch.ones(4, 1), batch_size=1)
                with redirect_stdout(io.StringIO()):
                    loss = train_epoch(model, loader, optimizer, scaler, device,
                                       self.args(accumulation=2), epoch=0,
                                       use_amp=device.type == "cuda")
                # Bad gradients invalidate the whole first accumulation group.
                # The second group must still update normally after zero_grad.
                self.assertEqual(len(optimizer.weights_before_step), 1)
                torch.testing.assert_close(optimizer.weights_before_step[0], original)
                self.assertEqual(scaler.get_scale(), 32768.0)
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                self.assertTrue(torch.isfinite(model.weight))
                self.assertLess(model.weight.item(), original.item())
                self.assertIsNone(model.weight.grad)

    def test_scaler_skips_overflow_and_recovers_on_cpu(self):
        self.check_scaler_recovery(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_fp16_scaler_skips_overflow_and_recovers_on_cuda(self):
        self.check_scaler_recovery(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_fp16_backward_overflow_without_injected_gradients(self):
        model = HalfPrecisionMAE().cuda()
        original = model.projection.weight.detach().clone()
        optimizer = CountingSGD(model.parameters())
        scaler = torch.amp.GradScaler("cuda", init_scale=65536.0)
        with redirect_stdout(io.StringIO()):
            loss = train_epoch(model, DataLoader(torch.ones(2, 1)), optimizer, scaler,
                               torch.device("cuda"), self.args(), epoch=0, use_amp=True)
        self.assertEqual(len(optimizer.weights_before_step), 1)
        torch.testing.assert_close(optimizer.weights_before_step[0], original)
        self.assertEqual(scaler.get_scale(), 32768.0)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertLess(model.projection.weight.item(), original.item())

    def test_unscaled_fp32_still_rejects_nonfinite_gradients(self):
        model = TinyMAE()
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        optimizer = CountingSGD(model.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            train_epoch(model, DataLoader(torch.ones(1, 1)), optimizer, scaler,
                        torch.device("cpu"), self.args(), epoch=0, use_amp=False)
        self.assertEqual(optimizer.weights_before_step, [])

    def test_nonfinite_loss_still_stops_before_any_update(self):
        for enabled in (False, True):
            with self.subTest(scaler_enabled=enabled):
                model = TinyMAE()
                optimizer = CountingSGD(model.parameters())
                scaler = torch.amp.GradScaler("cpu", enabled=enabled)
                with self.assertRaisesRegex(FloatingPointError, "Non-finite training loss"):
                    train_epoch(model, DataLoader(torch.full((1, 1), float("nan"))),
                                optimizer, scaler, torch.device("cpu"), self.args(),
                                epoch=0, use_amp=False)
                self.assertEqual(optimizer.weights_before_step, [])


if __name__ == "__main__":
    unittest.main()

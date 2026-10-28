"""Read-only gradient/update diagnostics for the four HF Swin stages."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch


def stage_parameters(backbone):
    groups = {}
    for index, layer in enumerate(backbone.encoder.layers, start=1):
        parameters = list(layer.parameters())
        parameters += list(backbone.hidden_states_norms[f"stage{index}"].parameters())
        if index == 1:
            parameters += list(backbone.embeddings.parameters())
        groups[f"stage{index}"] = parameters
    if len(groups) != 4:
        raise ValueError("Learning diagnostics require four Swin stages")
    return groups


class StageLearningMonitor:
    def __init__(self, backbone, output, logger, log_freq=20):
        if log_freq < 1:
            raise ValueError("stage_log_freq must be positive")
        self.groups = stage_parameters(backbone)
        self.path = Path(output) / "stage_learning.jsonl"
        self.logger, self.log_freq = logger, log_freq

    def begin_epoch(self, epoch):
        self.epoch = epoch
        self.attempts = self.skipped = 0
        # CPU snapshots cost RAM, not another backbone copy in GPU memory.
        self.before = {name: [p.detach().float().cpu().clone() for p in parameters]
                       for name, parameters in self.groups.items()}

    def wants_gradients(self, update, total_updates):
        return update == 1 or update % self.log_freq == 0 or update == total_updates

    def gradients(self):
        result = {}
        for name, parameters in self.groups.items():
            gradients = [p.grad.detach() for p in parameters if p.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([g.float().norm().double() for g in gradients])).item() if gradients else None
            result[name] = {
                "grad_norm": norm if norm is not None and math.isfinite(norm) else None,
                "grad_finite": math.isfinite(norm) if norm is not None else None,
                "parameters_with_grad": len(gradients),
                "parameters_without_grad": len(parameters) - len(gradients),
            }
        return result

    def write(self, record):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **record}, allow_nan=False) + "\n")

    def finish_update(self, update, total_updates, batch, gradients, scale_before, scale_after):
        self.attempts += 1
        skipped = scale_before is not None and scale_after < scale_before
        self.skipped += int(skipped)
        if gradients is not None:
            self.write({"kind": "gradients", "epoch": self.epoch, "update": update,
                        "total_updates": total_updates, "batch": batch, "optimizer_step_skipped": skipped,
                        "scale_before": scale_before, "scale_after": scale_after, "stages": gradients})
            for name, stats in gradients.items():
                norm = f"{stats['grad_norm']:.6e}" if stats["grad_norm"] is not None else "n/a"
                self.logger.info("[learning] epoch=%s update=%s/%s %s grad_norm=%s finite=%s grad_none=%s skipped=%s",
                                 self.epoch, update, total_updates, name, norm, stats["grad_finite"],
                                 stats["parameters_without_grad"], skipped)
        if skipped:
            self.logger.warning("[learning] epoch=%s update=%s AMP skipped update; scale=%s -> %s",
                                self.epoch, update, scale_before, scale_after)
            if gradients is None:
                self.write({"kind": "amp_skip", "epoch": self.epoch, "update": update, "batch": batch,
                            "optimizer_step_skipped": True, "scale_before": scale_before, "scale_after": scale_after})

    @torch.no_grad()
    def end_epoch(self):
        stages = {}
        for name, parameters in self.groups.items():
            squares_before = squares_delta = 0.0
            finite = True
            for parameter, previous in zip(parameters, self.before[name]):
                current = parameter.detach().float().cpu()
                finite = finite and bool(torch.isfinite(current).all())
                squares_before += previous.double().square().sum().item()
                squares_delta += (current.double() - previous.double()).square().sum().item()
            norm_before, delta = math.sqrt(squares_before), math.sqrt(squares_delta)
            finite = finite and math.isfinite(delta) and math.isfinite(norm_before)
            stats = {"parameter_norm_before": norm_before if finite else None,
                     "weight_delta_norm": delta if finite else None,
                     "weight_rel_change": delta / norm_before if finite and norm_before > 0 else None,
                     "changed": delta > 0 if finite else None, "finite": finite}
            stages[name] = stats
            relative = f"{stats['weight_rel_change']:.6e}" if stats["weight_rel_change"] is not None else "n/a"
            self.logger.info("[learning] epoch=%s %s weight_rel_change=%s changed=%s finite=%s",
                             self.epoch, name, relative, stats["changed"], finite)
        record = {"kind": "epoch_updates", "epoch": self.epoch, "attempted_updates": self.attempts,
                  "skipped_updates": self.skipped, "successful_updates": self.attempts - self.skipped,
                  "stages": stages}
        self.write(record)
        self.logger.info("[learning] epoch=%s successful_updates=%s/%s skipped=%s",
                         self.epoch, self.attempts - self.skipped, self.attempts, self.skipped)
        del self.before
        return record

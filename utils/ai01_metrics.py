"""Masked reconstruction metrics; evaluation only, with no extra dependencies.

Intensity errors use the original 0..255 reference scale. SSIM/PSNR use a fixed
0..1 data range after undoing dataset normalization; predictions are NOT clipped.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .ai01_targets import validate_target_z


METRICS = {
    "masked_psnr": {"unit": "dB", "better": "higher", "interpretation_vi": "Càng cao càng tốt"},
    "masked_ssim": {"unit": "dimensionless", "better": "higher", "interpretation_vi": "Càng gần 1 càng tốt"},
    "masked_mae": {"unit": "gray level (0..255 reference scale)", "better": "lower", "interpretation_vi": "Càng thấp càng tốt"},
    "masked_rmse": {"unit": "gray level (0..255 reference scale)", "better": "lower", "interpretation_vi": "Càng thấp càng tốt"},
    "masked_gradient_mae": {"unit": "gray-level difference per pixel", "better": "lower", "interpretation_vi": "Càng thấp càng tốt"},
}
METRIC_NAMES = tuple(METRICS)


def validate_ssim_window(window):
    if not isinstance(window, int) or window < 3 or window % 2 != 1:
        raise ValueError("ssim_window must be an odd integer >= 3")


@torch.no_grad()
def reconstruction_metrics(images, predictions, patch_mask, patch_size, preprocessing, ssim_window=7, target_z=None):
    """Return [B, selected Z, 5] metrics and [B] eligible SSIM window counts.

    SSIM uses uniform, valid windows and population covariance (K1=.01, K2=.03).
    Only windows ENTIRELY masked are eligible. Gradient MAE compares horizontal
    and vertical finite differences only when BOTH endpoints are masked.
    Undefined SSIM/gradient values are NaN internally and become JSON null with
    count=0 in reports. Exact reconstruction PSNR is capped at 120 dB.
    """
    validate_ssim_window(ssim_window)
    if images.ndim != 4 or predictions.shape != images.shape:
        raise ValueError("Metrics require matching [B, Z, H, W] images and predictions")
    batch, channels, height, width = images.shape
    if height < ssim_window or width < ssim_window:
        raise ValueError("ssim_window is larger than the input image")
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("Patch size must divide the metric image dimensions")
    if channels != preprocessing["num_channels"] or preprocessing["std"] <= 0:
        raise ValueError("Metric preprocessing does not match the images")
    target_z = validate_target_z(range(channels) if target_z is None else target_z, channels)
    expected = (batch, (height // patch_size) * (width // patch_size))
    if patch_mask.dtype != torch.bool or tuple(patch_mask.shape) != expected:
        raise ValueError(f"Expected boolean metric patch mask {expected}")
    if not bool(torch.isfinite(images[:, list(target_z)]).all() and torch.isfinite(predictions[:, list(target_z)]).all()):
        raise FloatingPointError("Non-finite image or reconstruction in metrics")
    pixel_mask = patch_mask.reshape(batch, 1, height // patch_size, width // patch_size)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
    count = pixel_mask.sum((-2, -1)).squeeze(1)
    if not bool((count > 0).all()):
        raise ValueError("Every metric image must have masked pixels")
    weights = pixel_mask.float()
    eligible = F.avg_pool2d(weights, ssim_window, stride=1, divisor_override=1) == ssim_window ** 2
    ssim_count = eligible.sum((-2, -1)).squeeze(1)
    horizontal = pixel_mask[..., :, 1:] & pixel_mask[..., :, :-1]
    vertical = pixel_mask[..., 1:, :] & pixel_mask[..., :-1, :]
    gradient_count = (horizontal.sum((-2, -1)) + vertical.sum((-2, -1))).squeeze(1)

    # Process one Z at a time to avoid several full B*11*H*W SSIM intermediates.
    planes = []
    with torch.autocast(device_type=images.device.type, enabled=False):
        for z in target_z:
            target = images[:, z:z + 1].float() * preprocessing["std"] + preprocessing["mean"]
            predicted = predictions[:, z:z + 1].float() * preprocessing["std"] + preprocessing["mean"]
            error = predicted - target
            mse = (error.square() * weights).sum((-2, -1)).squeeze(1) / count
            mae = (error.abs() * weights).sum((-2, -1)).squeeze(1) / count
            psnr = -10 * mse.clamp_min(1e-12).log10()

            mean_x = F.avg_pool2d(target, ssim_window, stride=1)
            mean_y = F.avg_pool2d(predicted, ssim_window, stride=1)
            var_x = (F.avg_pool2d(target.square(), ssim_window, stride=1) - mean_x.square()).clamp_min(0)
            var_y = (F.avg_pool2d(predicted.square(), ssim_window, stride=1) - mean_y.square()).clamp_min(0)
            covariance = F.avg_pool2d(target * predicted, ssim_window, stride=1) - mean_x * mean_y
            ssim_map = ((2 * mean_x * mean_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
                (mean_x.square() + mean_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2)
            )
            ssim = (ssim_map.clamp(-1, 1) * eligible).sum((-2, -1)).squeeze(1) / ssim_count

            gradient_error = (
                (error[..., :, 1:] - error[..., :, :-1]).abs() * horizontal
            ).sum((-2, -1)) + (
                (error[..., 1:, :] - error[..., :-1, :]).abs() * vertical
            ).sum((-2, -1))
            gradient_mae = gradient_error.squeeze(1) / gradient_count
            planes.append(torch.stack((psnr, ssim, mae * 255, mse.sqrt() * 255, gradient_mae * 255), -1))
    return torch.stack(planes, 1), ssim_count


def _mean(values, axis=0):
    """NaN-aware mean without RuntimeWarning for entirely undefined columns."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    result = np.full(np.shape(count), np.nan, dtype=np.float64)
    np.divide(np.where(valid, values, 0).sum(axis=axis), count, out=result, where=count > 0)
    return result


def _describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1, len(METRIC_NAMES))
    result = {}
    for index, name in enumerate(METRIC_NAMES):
        column = values[:, index]
        column = column[np.isfinite(column)]
        result[name] = {
            "mean": float(column.mean()) if len(column) else None,
            "std": float(column.std(ddof=0)) if len(column) else None,
            "count": int(len(column)),
        }
    return result


def _metric_text(stats):
    parts = []
    for name, definition in METRICS.items():
        entry = stats[name]
        value = "n/a" if entry["mean"] is None else f"{entry['mean']:.6f} +/- {entry['std']:.6f}"
        parts.append(f"{name}={value} ({definition['better']} is better; n={entry['count']})")
    return " | ".join(parts)


class ReconstructionMetricLogger:
    """Aggregate seeds -> Z -> frames -> wells, never treating frames as patients.

    Call update in ordered validation-loader order, then write once per epoch.
    One parent directory is assumed to be one well. Its parent name is reported
    as a directory group, NOT read/used as a supervised label.
    """

    def __init__(self, files, preprocessing, output=None, logger=None, ssim_window=7, context=None, target_z=None):
        validate_ssim_window(ssim_window)
        self.files = [Path(file) for file in files]
        if not self.files or len(set(self.files)) != len(self.files):
            raise ValueError("Metric files must be nonempty and unique")
        self.preprocessing = dict(preprocessing)
        self.target_z = validate_target_z(
            range(self.preprocessing["num_channels"]) if target_z is None else target_z,
            self.preprocessing["num_channels"],
        )
        self.output = Path(output) if output is not None else None
        self.logger, self.ssim_window = logger, ssim_window
        self.context = dict(context or {})
        self.values = defaultdict(list)
        self.pairs = set()
        self.ssim_windows = []

    def update(self, images, predictions, mask, patch_size, start_index, seed):
        if start_index < 0 or start_index + len(images) > len(self.files):
            raise ValueError("Metric batch does not match the validation manifest")
        values, counts = reconstruction_metrics(
            images, predictions, mask, patch_size, self.preprocessing, self.ssim_window, self.target_z,
        )
        values = values.detach().double().cpu().numpy()
        for offset, value in enumerate(values):
            index = start_index + offset
            pair = (index, seed)
            if pair in self.pairs:
                raise ValueError("Duplicate validation image/mask-seed pair")
            self.pairs.add(pair)
            self.values[index].append(value)
        self.ssim_windows.extend(counts.cpu().tolist())

    def summary(self, epoch):
        seeds = sorted({seed for _, seed in self.pairs})
        expected = {(index, seed) for index in range(len(self.files)) for seed in seeds}
        if not seeds or self.pairs != expected:
            raise ValueError("Metrics require every validation image for every mask seed")
        # Arithmetic mean over seeds, independently for each frame and Z.
        frame_z = np.stack([_mean(self.values[i]) for i in range(len(self.files))])
        frame = _mean(frame_z, axis=1)
        wells = defaultdict(list)
        for index, file in enumerate(self.files):
            wells[file.parent.as_posix()].append(index)
        well_z = np.stack([_mean(frame_z[indices]) for indices in wells.values()])
        well = _mean(well_z, axis=1)
        groups = defaultdict(list)
        per_well = {}
        for well_index, (name, indices) in enumerate(wells.items()):
            group = Path(name).parent.name or "unknown"
            groups[group].append(well_index)
            per_well[name] = {"group": group, "num_frames": len(indices), "frame_macro": _describe(frame[indices])}
        report = {
            "format": "ai01-reconstruction-metrics-v1", "time_utc": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch, "num_frames": len(self.files), "num_wells": len(wells),
            "num_z": int(frame_z.shape[1]), "num_evaluations": len(self.pairs), "mask_seeds": seeds,
            "input_num_z": self.preprocessing["num_channels"], "target_z": list(self.target_z),
            "context": self.context, "metric_definitions": METRICS,
            "protocol": {
                "intensity_transform": "x_01 = normalized_x * preprocessing.std + preprocessing.mean",
                "preprocessing": self.preprocessing, "predictions_clipped": False, "data_range": 1.0,
                "psnr_mse_floor": 1e-12, "psnr_cap_db": 120,
                "ssim_window": self.ssim_window, "ssim_weights": "uniform", "ssim_covariance": "population",
                "ssim_region": "valid windows entirely inside the masked region; no visible/copied pixels",
                "gradient_region": "horizontal/vertical pairs with both endpoints masked",
                "undefined_metrics": "JSON null with count=0; excluded from aggregates, never replaced by zero",
                "aggregation": "mean seeds per frame/Z; mean selected target Z per frame; mean frames per well; equal-weight mean wells",
                "std": "population standard deviation across the reported units; not a confidence interval",
                "psnr_aggregation": "arithmetic mean of per-frame/per-Z/per-seed dB, NOT PSNR of pooled MSE",
                "mask_reproducibility": "fixed seed, ordered manifest, batch size and device; keep these fixed for comparisons",
                "group_definition": "directory name above each well; not supervised labels",
            },
            "ssim_eligible_windows_per_evaluation": {
                "min": int(min(self.ssim_windows)), "max": int(max(self.ssim_windows)),
                "evaluations_without_windows": sum(count == 0 for count in self.ssim_windows),
            },
            "frame_macro": _describe(frame), "well_macro": _describe(well),
            "per_z": {f"Z{z:02d}": {"frame_macro": _describe(frame_z[:, index]), "well_macro": _describe(well_z[:, index])}
                      for index, z in enumerate(self.target_z)},
            "per_well": per_well,
            "per_group": {},
        }
        well_names = list(wells)
        for group, well_indices in groups.items():
            indices = [i for w in well_indices for i in wells[well_names[w]]]
            report["per_group"][group] = {
                "num_frames": len(indices), "num_wells": len(well_indices),
                "frame_macro": _describe(frame[indices]), "well_macro": _describe(well[well_indices]),
            }
        return report

    def write(self, epoch):
        report = self.summary(epoch)
        lines = [
            f"[reconstruction] epoch={epoch} frames={report['num_frames']} wells={report['num_wells']} "
            f"input_Z={report['input_num_z']} target_z={report['target_z']} mask_seeds={report['mask_seeds']}",
            "Context: " + json.dumps(report["context"], ensure_ascii=True),
            "Directions: PSNR higher is better; SSIM closer to 1 is better; MAE/RMSE/gradient MAE lower is better.",
            "Huong doc: " + "; ".join(f"{name}: {definition['interpretation_vi']}" for name, definition in METRICS.items()),
            "Region: masked pixels only; SSIM uses fully masked valid windows; no clipping.",
            "Units: PSNR dB; SSIM dimensionless; MAE/RMSE/gradient MAE on the 0..255 reference scale.",
            "Mean +/- population std; well_macro is the primary equal-weight-well summary, not a confidence interval.",
            "frame_macro | " + _metric_text(report["frame_macro"]),
            "well_macro  | " + _metric_text(report["well_macro"]),
        ]
        for z, stats in report["per_z"].items():
            lines.append(z + " well_macro | " + _metric_text(stats["well_macro"]))
        for group, stats in report["per_group"].items():
            lines.append(f"group={group} frames={stats['num_frames']} wells={stats['num_wells']} well_macro | " + _metric_text(stats["well_macro"]))
        for name, stats in report["per_well"].items():
            lines.append(f"well={name} frames={stats['num_frames']} frame_macro | " + _metric_text(stats["frame_macro"]))
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / "reconstruction_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
            temporary = self.output / "reconstruction_metrics_latest.json.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            temporary.replace(self.output / "reconstruction_metrics_latest.json")
            with (self.output / "reconstruction_metrics.log").open("a", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n\n")
        if self.logger is not None:
            self.logger.info(lines[0])
            self.logger.info("[reconstruction] frame_macro | %s", _metric_text(report["frame_macro"]))
            self.logger.info("[reconstruction] well_macro | %s", _metric_text(report["well_macro"]))
            if report["ssim_eligible_windows_per_evaluation"]["evaluations_without_windows"]:
                self.logger.warning("[reconstruction] Some masks have no fully masked SSIM windows; see counts/nulls in the report")
        return report

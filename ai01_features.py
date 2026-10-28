"""Fixed-validation-sample diagnostics; never participates in optimization."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.swin.modeling_swin import SwinLayer


def select_feature_indices(files, count):
    """Prefer one frame per well, then fill from the remaining sorted frames."""
    selected, seen_wells = [], set()
    for index, file in enumerate(files):
        well = Path(file).parent
        if well not in seen_wells:
            selected.append(index)
            seen_wells.add(well)
    selected_set = set(selected)
    selected.extend(index for index in range(len(files)) if index not in selected_set)
    return selected[:count]


def feature_statistics(feature):
    feature = feature.detach().float().cpu()
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("Feature diagnostics expect [1, C, H, W]")
    nonfinite_count = int((~torch.isfinite(feature)).sum().item())
    stats = {"shape": list(feature.shape), "finite": nonfinite_count == 0,
             "nonfinite_count": nonfinite_count}
    # Keep JSON valid even when diagnostics uncover NaN/Inf.
    if nonfinite_count:
        stats.update(dict.fromkeys(("min", "max", "mean", "std", "spatial_std")))
    else:
        stats.update({
            "min": feature.min().item(), "max": feature.max().item(),
            "mean": feature.mean().item(), "std": feature.std(unbiased=False).item(),
            "spatial_std": feature.flatten(2).std(dim=-1, unbiased=False).mean().item(),
        })
    return stats


def feature_geometry(feature):
    """Exact centered PCA on tokens x channels; no random projection/subsampling."""
    tokens = feature[0].flatten(1).T.double()
    count, channels = tokens.shape
    maximum_rank = min(max(0, count - 1), channels)
    keys = ("effective_rank", "effective_rank_fraction", "pca_top1_variance_ratio",
            "pca_top3_variance_ratio", "mean_token_cosine", "low_variance_channel_fraction",
            "spatial_std_min", "spatial_std_p10", "spatial_std_median", "spatial_std_max",
            "zero_token_fraction")
    stats = {"max_centered_rank": maximum_rank, "low_variance_std_threshold": 1e-3,
             **dict.fromkeys(keys)}
    if not bool(torch.isfinite(tokens).all()):
        return stats, None
    spatial_std = tokens.std(dim=0, unbiased=False)
    quantiles = spatial_std.quantile(torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float64))
    stats.update(dict(zip(("spatial_std_min", "spatial_std_p10", "spatial_std_median", "spatial_std_max"), quantiles.tolist())))
    stats["low_variance_channel_fraction"] = (spatial_std < 1e-3).double().mean().item()
    norms = tokens.norm(dim=1)
    valid = norms > 1e-12
    stats["zero_token_fraction"] = (~valid).double().mean().item()
    unit = tokens[valid] / norms[valid, None]
    if len(unit) > 1:
        cosine = (unit.sum(0).square().sum() - unit.square().sum()) / (len(unit) * (len(unit) - 1))
        stats["mean_token_cosine"] = cosine.clamp(-1, 1).item()
    mean = tokens.mean(dim=0)
    centered = tokens - mean
    _, singular, vectors = torch.linalg.svd(centered, full_matrices=False)
    variance = singular.square()
    total = variance.sum().item()
    components = torch.zeros(channels, 3, dtype=torch.float64)
    if total > 0 and maximum_rank:
        probability = variance / total
        positive = probability[probability > 0]
        rank = min(float(positive.mul(positive.log()).sum().neg().exp()), maximum_rank)
        stats.update(effective_rank=rank, effective_rank_fraction=rank / maximum_rank,
                     pca_top1_variance_ratio=probability[0].item(),
                     pca_top3_variance_ratio=probability[:3].sum().clamp(max=1).item())
        count_components = min(3, maximum_rank)
        components[:, :count_components] = vectors[:count_components].T
        # Fix a deterministic sign convention; bases/limits are then frozen.
        for column in range(count_components):
            anchor = components[:, column].abs().argmax()
            if components[anchor, column] < 0:
                components[:, column].neg_()
    else:
        stats.update(effective_rank=0.0, effective_rank_fraction=0.0)
    projection = centered @ components
    limits = projection.quantile(torch.tensor([0.01, 0.99], dtype=torch.float64), dim=0)
    basis = {"mean": mean, "components": components, "low": limits[0], "high": limits[1]}
    return stats, basis


def pca_rgb(feature, basis):
    height, width = feature.shape[-2:]
    if basis is None or not bool(torch.isfinite(feature).all()):
        return np.full((height, width, 3), 0.5)
    tokens = feature[0].flatten(1).T.double()
    projected = (tokens - basis["mean"]) @ basis["components"]
    span = basis["high"] - basis["low"]
    colors = (projected - basis["low"]) / span.clamp_min(1e-12)
    colors[:, span <= 1e-12] = 0.5
    return colors.clamp(0, 1).reshape(height, width, 3).numpy()


def feature_drift(feature, baseline):
    current, previous = feature.double().flatten(), baseline.double().flatten()
    if not bool(torch.isfinite(current).all() and torch.isfinite(previous).all()):
        return {"feature_rel_l2_from_baseline": None, "feature_cosine_from_baseline": None}
    current_norm, previous_norm = current.norm().item(), previous.norm().item()
    return {
        "feature_rel_l2_from_baseline": (current - previous).norm().item() / previous_norm if previous_norm > 0 else None,
        "feature_cosine_from_baseline": (torch.dot(current, previous) / (current_norm * previous_norm)).clamp(-1, 1).item()
        if current_norm > 0 and previous_norm > 0 else None,
    }


def save_pca_grid(features, references, path, record):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, 4.5), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)
    fig.suptitle(f"PCA FEATURES | epoch {record['epoch']} | val {record['sample_index']} | {record['mode']}", fontsize=15)
    for index, (feature, reference, axis) in enumerate(zip(features, references, axes), start=1):
        axis.imshow(pca_rgb(feature, reference["pca"]), interpolation="nearest")
        stats = record["stages"][f"stage{index}"]
        rank = f"{stats['effective_rank']:.2f}" if stats["effective_rank"] is not None else "n/a"
        axis.set_title(f"Stage {index} | {feature.shape[-2]} x {feature.shape[-1]}\neffective rank={rank}")
        axis.axis("off")
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.02, right=0.98, wspace=0.1)
    fig.text(0.5, 0.1, f"Projection + color limits frozen from baseline epoch {record['baseline_epoch']} ({record['baseline_origin']}).",
             ha="center", fontsize=10)
    fig.text(0.5, 0.04, "PCA colors are not PN labels. Structure/change alone does not prove useful learning.", ha="center", fontsize=10)
    try:
        temporary = path.with_suffix(".png.tmp")
        fig.savefig(temporary, format="png")
        temporary.replace(path)
    finally:
        fig.clear()


def unmasked_features(backbone, images):
    # HF's ordinary forward can mutate shift/window sizes on small stage maps
    # (notably stage4=7x7 at input 224). Debugging must not alter later SSL passes.
    runtime = [(module, module.shift_size, module.window_size)
               for module in backbone.modules() if isinstance(module, SwinLayer)]
    try:
        return backbone(images).feature_maps
    finally:
        for module, shift_size, window_size in runtime:
            module.shift_size, module.window_size = shift_size, window_size


def save_feature_grid(features, path, record):
    # Use the non-interactive canvas directly; safe on Colab/headless servers.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 11), dpi=150, facecolor="#f8fafc")
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 4)
    fig.suptitle(
        f"SWIN FEATURE MAPS | epoch {record['epoch']} | val index {record['sample_index']}",
        fontsize=16, fontweight="bold", y=0.975,
    )
    fig.text(0.5, 0.94,
             f"mode={record['mode']} | input={record['input_shape']} | "
             f"mask seed={record['mask_seed']} | eval / FP32",
             ha="center", fontsize=10)
    source = Path(record["source_image"])
    fig.text(0.5, 0.91, f"Source: {source.parent.name}/{source.name}", ha="center", fontsize=9)
    for row, feature in enumerate(features):
        channels, height, width = feature.shape[1:]
        for column, channel in enumerate(record["stages"][f"stage{row + 1}"]["channels_shown"]):
            axis = axes[row, column]
            view = feature[0, channel].numpy()
            finite_values = view[np.isfinite(view)]
            low, high = np.quantile(finite_values, [0.01, 0.99]) if finite_values.size else (0.0, 1.0)
            axis.imshow(np.ma.masked_invalid(view), cmap="viridis", interpolation="nearest",
                        vmin=low, vmax=max(high, low + 1e-8))
            axis.set_title(f"Channel {channel} | {height} x {width}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(f"STAGE {row + 1}\n{channels} channels", rotation=0,
                                labelpad=50, va="center", fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.075, hspace=0.34, wspace=0.15)
    fig.text(0.5, 0.035, "Each panel uses its own 1st-99th percentile contrast; colors are activations, not PN labels.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.017, "Feature-map appearance alone does not demonstrate useful learning.", ha="center", fontsize=9)
    temporary = path.with_suffix(".png.tmp")
    try:
        fig.savefig(temporary, format="png", facecolor=fig.get_facecolor())
        temporary.replace(path)
    finally:
        fig.clear()


class FeatureLogger:
    def __init__(self, model, dataset, output, logger, samples=3, image_freq=5,
                  mode="both", seed=42):
        if samples < 1 or image_freq < 0 or mode not in ("masked", "unmasked", "both"):
            raise ValueError("Invalid feature diagnostic settings")
        if getattr(dataset, "augment", False):
            raise ValueError("Feature diagnostics require a dataset without augmentation")
        if image_freq:
            # Fail before training, not at the first epoch's PNG save.
            from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
        self.output = Path(output)
        self.directory = self.output / "feature_debug"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger, self.image_freq = logger, image_freq
        self.modes = ("masked", "unmasked") if mode == "both" else (mode,)
        self.samples = []
        self.references = {}
        self.baseline_epoch, self.baseline_origin = None, None
        for index in select_feature_indices(dataset.files, samples):
            image = dataset[index].unsqueeze(0).detach().cpu()
            mask_seed = seed + index
            # CPU generator + cached mask: independent of train/val RNG and GPU.
            mask = model.make_mask(1, "cpu", torch.Generator().manual_seed(mask_seed))
            metadata = {
                "sample_index": index, "source_image": str(dataset.files[index]),
                "input_shape": list(image.shape), "mask_seed": mask_seed,
                "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                "image_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                "masked_fraction": mask.float().mean().item(),
            }
            self.samples.append((image, mask, metadata))
        (self.directory / "samples.json").write_text(
            json.dumps({"modes": list(self.modes), "samples": [item[2] for item in self.samples]}, indent=2),
            encoding="utf-8",
        )
        logger.info("[features] Fixed validation indices=%s modes=%s; statistics every epoch; image_freq=%s",
                    [item[2]["sample_index"] for item in self.samples], mode, image_freq)
        signature = hashlib.sha256(json.dumps({"modes": self.modes, "samples": [item[2] for item in self.samples]},
                                             sort_keys=True).encode()).hexdigest()[:16]
        self.baseline_path = self.directory / f"baseline_{signature}.pt"

    def initialize_baseline(self, model, device, epoch, total_epochs, origin, resume_directory=None):
        candidate = self.baseline_path
        if resume_directory is not None and not candidate.is_file():
            candidate = Path(resume_directory) / "feature_debug" / self.baseline_path.name
        if origin == "resume" and candidate.is_file():
            state = torch.load(candidate, map_location="cpu", weights_only=True)
            if state.get("format") != "ai01-feature-baseline-v1":
                raise ValueError("Invalid feature baseline format")
            self.references = state["references"]
            self.baseline_epoch, self.baseline_origin = state["epoch"], state["origin"]
            self.logger.info("[features] Loaded fixed baseline epoch=%s origin=%s from %s",
                             self.baseline_epoch, self.baseline_origin, candidate)
            if candidate != self.baseline_path:
                self.save_baseline()
            return
        if origin != "resume" and self.baseline_path.exists():
            raise FileExistsError("Feature baseline already exists; use a new output directory or resume")
        self.baseline_epoch, self.baseline_origin = epoch, origin
        self.logger.info("[features] Creating baseline epoch=%s origin=%s", epoch, origin)
        self.log_epoch(model, device, epoch, total_epochs, baseline=True)

    def save_baseline(self):
        temporary = self.baseline_path.with_suffix(".pt.tmp")
        torch.save({"format": "ai01-feature-baseline-v1", "epoch": self.baseline_epoch,
                    "origin": self.baseline_origin, "references": self.references}, temporary)
        temporary.replace(self.baseline_path)

    def save_images_at(self, epoch, total_epochs):
        return self.image_freq > 0 and (epoch in (0, 1) or epoch % self.image_freq == 0 or epoch == total_epochs)

    @torch.inference_mode()
    def log_epoch(self, model, device, epoch, total_epochs, baseline=False):
        creating_reference = not self.references
        if self.baseline_epoch is None:
            self.baseline_epoch, self.baseline_origin = epoch, "diagnostic_start"
        was_training = model.training
        model.eval()
        try:
            for image, mask, metadata in self.samples:
                image = image.to(device)
                for mode in self.modes:
                    # No autocast: use a consistent FP32 diagnostic pass.
                    if mode == "masked":
                        features = model.forward_encoder(image, mask.to(device))
                    else:
                        features = unmasked_features(model.backbone, image)
                    if len(features) != 4:
                        raise ValueError("Feature diagnostics require all four Swin stages")
                    features = tuple(feature.detach().float().cpu() for feature in features)
                    record = {
                        "time_utc": datetime.now(timezone.utc).isoformat(), "epoch": epoch,
                        **metadata, "mode": mode, "precision": "float32", "stages": {},
                        "baseline_epoch": self.baseline_epoch, "baseline_origin": self.baseline_origin,
                        "kind": "baseline" if baseline else "epoch",
                    }
                    if mode == "unmasked":
                        record.update(mask_seed=None, mask_sha256=None, masked_fraction=0.0)
                    self.logger.info("[features] epoch=%s sample=%s mode=%s mask_seed=%s source=%s",
                                     epoch, record["sample_index"], mode, record["mask_seed"], record["source_image"])
                    for index, feature in enumerate(features, start=1):
                        name = f"stage{index}"
                        stats = feature_statistics(feature)
                        geometry, fitted_basis = feature_geometry(feature)
                        stats.update(geometry)
                        key = f"{metadata['sample_index']}:{mode}:{name}"
                        if key not in self.references:
                            self.references[key] = {"feature": feature.clone(), "pca": fitted_basis}
                        stats.update(feature_drift(feature, self.references[key]["feature"]))
                        channels = feature.shape[1]
                        stats["channels_shown"] = [0, channels // 3, 2 * channels // 3, channels - 1]
                        record["stages"][name] = stats
                        values = " ".join(f"{key}={stats[key]:.6f}" if stats[key] is not None else f"{key}=n/a"
                                          for key in ("min", "max", "mean", "std", "spatial_std"))
                        self.logger.info("[features] %s shape=%s finite=%s %s", name, tuple(feature.shape), stats["finite"], values)
                        values = " ".join(f"{key}={stats[key]:.6f}" if stats[key] is not None else f"{key}=n/a"
                                          for key in ("effective_rank", "effective_rank_fraction", "pca_top1_variance_ratio",
                                                      "pca_top3_variance_ratio", "mean_token_cosine", "low_variance_channel_fraction",
                                                      "feature_rel_l2_from_baseline", "feature_cosine_from_baseline"))
                        self.logger.info("[features] %s %s", name, values)
                        if not stats["finite"]:
                            self.logger.warning("[features] %s has %s non-finite activations", name, stats["nonfinite_count"])
                    with (self.output / "feature_stats.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    if self.save_images_at(epoch, total_epochs) or baseline and self.image_freq:
                        prefix = "baseline_resume_" if baseline and self.baseline_origin == "resume" else ""
                        path = self.directory / f"{prefix}epoch{epoch:03d}_val{metadata['sample_index']:04d}_{mode}.png"
                        save_feature_grid(features, path, record)
                        self.logger.info("[features] Saved %s", path.relative_to(self.output).as_posix())
                        references = [self.references[f"{metadata['sample_index']}:{mode}:stage{i}"] for i in range(1, 5)]
                        pca_path = path.with_name(path.stem + "_pca.png")
                        save_pca_grid(features, references, pca_path, record)
                        self.logger.info("[features] Saved %s", pca_path.relative_to(self.output).as_posix())
            if creating_reference:
                self.save_baseline()
        finally:
            model.train(was_training)

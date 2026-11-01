"""Evaluate a full AI01 SSL checkpoint on a validation manifest without training."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch
from torch.utils.data import DataLoader

from ai01_data import AI01UnlabelledDataset
from ai01_metrics import ReconstructionMetricLogger, validate_ssim_window
from ai01_swin_mae import AI01SwinMAE
from ai01_targets import checkpoint_model_config, validate_target_z
from ai01_transfer import validate_preprocessing
from train_ai01 import configure_logger, dataset_signature, validate


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True, help="Full checkpoint-last.pt, NOT encoder-only export")
    result.add_argument("--val_manifest", required=True)
    result.add_argument("--output_dir", required=True, help="New empty directory for evaluation logs")
    result.add_argument("--batch_size", type=int, default=4)
    result.add_argument("--num_workers", type=int, default=0)
    result.add_argument("--array_layout", choices=["auto", "zhwc", "zhw", "hwz"], default="auto")
    result.add_argument("--mask_seeds", nargs="+", type=int, help="Defaults to the checkpoint's validation seed")
    result.add_argument("--target_z", nargs="+", type=int,
                        help="Optional evaluation subset of the checkpoint's supervised Z; defaults to saved targets")
    result.add_argument("--ssim_window", type=int, default=7)
    result.add_argument("--log_freq", type=int, default=20)
    result.add_argument("--max_samples", type=int, default=0, help="0=ALL validation images; nonzero is a labeled subset/smoke test")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--no_amp", action="store_true")
    return result


def main(args):
    validate_ssim_window(args.ssim_window)
    if args.batch_size < 1 or args.num_workers < 0 or args.log_freq < 1 or args.max_samples < 0:
        raise ValueError("Require batch_size/log_freq > 0 and num_workers/max_samples >= 0")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Select an available CPU or CUDA device")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "ai01-swin-mae-training-v1":
        raise ValueError("Reconstruction needs a FULL SSL checkpoint, including decoder; encoder-only exports cannot be evaluated")
    model_config = checkpoint_model_config(state["model_config"])
    checkpoint_target_z = list(model_config["target_z"])
    if args.target_z is not None:
        selected = validate_target_z(args.target_z, state["preprocessing"]["num_channels"])
        if not set(selected).issubset(checkpoint_target_z):
            raise ValueError("Evaluation target_z includes Z not supervised by this checkpoint")
        model_config["target_z"] = list(selected)
    if model_config.get("norm_pix_loss", False):
        raise ValueError("Absolute-intensity metrics are undefined for norm_pix_loss outputs; do not use ground-truth patch statistics to invert them")
    preprocessing = dict(state["preprocessing"])
    validate_preprocessing(preprocessing)
    if preprocessing["image_size"] != model_config["image_size"] or args.ssim_window > model_config["image_size"]:
        raise ValueError("Checkpoint image size, preprocessing or SSIM window mismatch")
    seeds = args.mask_seeds if args.mask_seeds is not None else [state["args"].get("seed", 42)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("mask_seeds must be distinct")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output directory must be empty; never overwrite training logs")
    dataset = AI01UnlabelledDataset(manifest=args.val_manifest, preprocessing=preprocessing,
                                   array_layout=args.array_layout, augment=False)
    original_count = len(dataset)
    if args.max_samples:
        dataset.files = dataset.files[:args.max_samples]
    model = AI01SwinMAE(**model_config)
    model.load_state_dict(state["model"], strict=True)
    epoch = state["epoch"]
    del state  # Release optimizer state before allocating GPU memory.
    model.to(device)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output)
    use_amp = device.type == "cuda" and not args.no_amp
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        generator=torch.Generator().manual_seed(0))
    context = {
        "scope": "full_validation" if len(dataset) == original_count else "first_n_samples_smoke_test",
        "checkpoint": str(Path(args.checkpoint).resolve()), "manifest": str(Path(args.val_manifest).resolve()),
        "original_manifest_frames": original_count, "batch_size": args.batch_size, "device": str(device),
        "amp": use_amp, "dataset_signature": dataset_signature(dataset),
        "checkpoint_target_z": checkpoint_target_z, "evaluation_target_z": list(model.target_z),
    }
    monitor = ReconstructionMetricLogger(dataset.files, preprocessing, output, logger, args.ssim_window, context,
                                        target_z=model.target_z)
    losses = {}
    logger.info("Evaluation only: epoch=%s frames=%s/%s mask_seeds=%s scope=%s", epoch, len(dataset), original_count, seeds, context["scope"])
    logger.info("Input: all 11 Z; checkpoint supervised Z=%s; evaluating Z=%s", checkpoint_target_z, list(model.target_z))
    for seed in seeds:
        logger.info("Evaluating mask seed=%s over ALL %s selected frames", seed, len(dataset))
        losses[str(seed)] = validate(model, loader, device, seed, use_amp, epoch=epoch - 1,
                                    log_freq=args.log_freq, logger=logger, step_log=output / "steps.jsonl",
                                    reconstruction_monitor=monitor)
    monitor.context["validation_loss_by_seed"] = losses
    report = monitor.write(epoch)
    logger.info("Evaluation complete: %s", output / "reconstruction_metrics_latest.json")
    return report


if __name__ == "__main__":
    main(parser().parse_args())

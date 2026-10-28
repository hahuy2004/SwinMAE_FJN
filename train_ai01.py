"""Train AI01's 11-channel HF Swin-T using unlabelled Z-stacks only."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

# This entry point is PyTorch-only; do not import an unrelated TensorFlow install.
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai01_data import AI01UnlabelledDataset, DEFAULT_PREPROCESSING, ensure_disjoint_wells
from ai01_swin_mae import AI01SwinMAE
from ai01_transfer import export_encoder, load_ai01_encoder


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--data_path", help="Prepared IMAGE directory only; recursively reads .npy")
    source.add_argument("--manifest", help="Text file listing unlabelled training .npy images")
    result.add_argument("--val_manifest", help="Optional fixed SSL validation set from separate wells")
    result.add_argument("--array_layout", choices=["auto", "zhwc", "zhw", "hwz"], default="auto")
    result.add_argument("--image_size", type=int, choices=[224, 256], default=256,
                        help="Square center-crop and SSL reconstruction size; patch size stays 4")
    result.add_argument("--sampling", choices=["well", "frame"], default="well")
    result.add_argument("--no_augment", action="store_true")
    result.add_argument("--output_dir", default="output_ai01_ssl")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch_size", type=int, default=4)
    result.add_argument("--accum_iter", type=int, default=1)
    result.add_argument("--num_workers", type=int, default=0)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--min_lr", type=float, default=1e-6)
    result.add_argument("--weight_decay", type=float, default=0.05)
    result.add_argument("--warmup_epochs", type=int, default=10)
    result.add_argument("--clip_grad", type=float, default=1.0)
    result.add_argument("--mask_ratio", type=float, default=0.75)
    result.add_argument("--mask_group_size", type=int, default=4)
    result.add_argument("--decoder_depth", type=int, default=2)
    result.add_argument("--norm_pix_loss", action="store_true")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--no_amp", action="store_true")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--save_freq", type=int, default=1)
    result.add_argument("--log_freq", type=int, default=20,
                        help="Log train/validation every N batches, plus the first and last batch")
    result.add_argument("--max_steps", type=int, default=0, help="Limit batches PER EPOCH; 0=all (smoke tests)")
    result.add_argument("--resume", help="checkpoint-last.pt: full SSL state at an epoch boundary")
    result.add_argument("--init_encoder", help="Optional encoder export to continue SSL, not RGB weights")
    return result


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def set_learning_rate(optimizer, args, epoch, step, steps):
    progress = epoch + (step + 1) / steps
    if args.warmup_epochs and progress < args.warmup_epochs:
        lr = args.lr * progress / args.warmup_epochs
    else:
        fraction = (progress - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = lr


def autocast_context(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def configure_logger(output):
    logger = logging.getLogger("ai01_ssl")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(Path(output) / "train.log", mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_progress(record, step_log=None, logger=None):
    message = (f"[{record['phase']}] epoch={record['epoch']} "
               f"step={record['step']}/{record['total_steps']} "
               f"loss={record['loss']:.6f} avg_loss={record['loss_avg']:.6f}")
    if "lr" in record:
        message += f" lr={record['lr']:.3e}"
    message += f" elapsed={record['elapsed_seconds']:.1f}s eta={record['eta_seconds']:.1f}s"
    if logger is None:
        print(message, flush=True)
    else:
        logger.info(message)
    if step_log is not None:
        with Path(step_log).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **record}) + "\n")


def train_epoch(model, loader, optimizer, scaler, device, args, epoch, use_amp,
                logger=None, step_log=None):
    model.train()
    steps = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    total, samples = 0.0, 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for index, images in enumerate(islice(loader, steps)):
        set_learning_rate(optimizer, args, epoch, index, steps)
        images = images.to(device, non_blocking=device.type == "cuda")
        group_start = (index // args.accum_iter) * args.accum_iter
        group_size = min(args.accum_iter, steps - group_start)
        with autocast_context(device, use_amp):
            loss, _, _ = model(images)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, batch {index}")
        scaler.scale(loss / group_size).backward()
        if (index + 1) % args.accum_iter == 0 or index + 1 == steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * images.shape[0]
        samples += images.shape[0]
        if index == 0 or (index + 1) % args.log_freq == 0 or index + 1 == steps:
            elapsed = time.perf_counter() - started
            log_progress({
                "phase": "train", "epoch": epoch + 1, "step": index + 1,
                "total_steps": steps, "global_step": epoch * steps + index + 1,
                "loss": loss.item(), "loss_avg": total / samples,
                "lr": optimizer.param_groups[0]["lr"], "samples_seen": samples,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / (index + 1) * (steps - index - 1),
            }, step_log, logger)
    return total / samples


@torch.no_grad()
def validate(model, loader, device, seed, use_amp, epoch=0, log_freq=20,
             logger=None, step_log=None):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, samples = 0.0, 0
    started = time.perf_counter()
    for index, images in enumerate(loader):
        images = images.to(device)
        mask = model.make_mask(images.shape[0], device, generator)
        with autocast_context(device, use_amp):
            loss, _, _ = model(images, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SSL validation loss")
        total += loss.item() * images.shape[0]
        samples += images.shape[0]
        if index == 0 or (index + 1) % log_freq == 0 or index + 1 == len(loader):
            elapsed = time.perf_counter() - started
            log_progress({
                "phase": "validation", "epoch": epoch + 1, "step": index + 1,
                "total_steps": len(loader), "loss": loss.item(), "loss_avg": total / samples,
                "samples_seen": samples, "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / (index + 1) * (len(loader) - index - 1),
            }, step_log, logger)
    return total / samples


def atomic_save(payload, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def dataset_signature(dataset):
    return hashlib.sha256("\n".join(map(str, dataset.files)).encode("utf-8")).hexdigest()


def main(args):
    if args.image_size not in (224, 256):
        raise ValueError("image_size must be 224 or 256")
    for name in ("epochs", "batch_size", "accum_iter", "save_freq", "log_freq"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0 or args.max_steps < 0 or args.warmup_epochs < 0:
        raise ValueError("Worker, step and warmup counts cannot be negative")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0 or args.clip_grad <= 0:
        raise ValueError("Require lr > 0, 0 <= min_lr <= lr and clip_grad > 0")
    if args.resume and args.init_encoder:
        raise ValueError("Use resume OR init_encoder, not both")
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("This entry point supports CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; pass --device cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output)
    step_log = output / "steps.jsonl"
    use_amp = device.type == "cuda" and not args.no_amp
    logger.info("Device=%s AMP=%s batch_size=%s accumulation=%s log_freq=%s batches",
                device, use_amp, args.batch_size, args.accum_iter, args.log_freq)
    logger.info("Output directory: %s", output.resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    preprocessing = dict(DEFAULT_PREPROCESSING)
    preprocessing["image_size"] = args.image_size
    dataset = AI01UnlabelledDataset(
        data_path=args.data_path, manifest=args.manifest, preprocessing=preprocessing,
        array_layout=args.array_layout, augment=not args.no_augment,
    )
    # Fail early on format errors, before allocating the model/GPU.
    sample = dataset[0]
    logger.info("Images=%s, first tensor=%s, sampling=%s", len(dataset), tuple(sample.shape), args.sampling)
    if args.manifest:
        logger.info("Train manifest: %s (relative image paths resolve from this file's parent)",
                    Path(args.manifest).resolve())
    sampler = dataset.well_balanced_sampler(generator) if args.sampling == "well" else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
        generator=generator, worker_init_fn=seed_worker,
    )
    validation_loader = None
    validation_dataset = None
    if args.val_manifest:
        validation_dataset = AI01UnlabelledDataset(
            manifest=args.val_manifest, preprocessing=preprocessing,
            array_layout=args.array_layout, augment=False,
        )
        ensure_disjoint_wells(dataset, validation_dataset)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        logger.info("Validation images=%s, manifest=%s", len(validation_dataset), Path(args.val_manifest).resolve())
    model = AI01SwinMAE(
        mask_ratio=args.mask_ratio, mask_group_size=args.mask_group_size,
        norm_pix_loss=args.norm_pix_loss, decoder_depth=args.decoder_depth,
        image_size=args.image_size,
    )
    if args.init_encoder:
        logger.info("Initialized encoder: %s", load_ai01_encoder(model.backbone, args.init_encoder))
    model.to(device)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    signature = dataset_signature(dataset)
    validation_signature = dataset_signature(validation_dataset) if validation_dataset else None
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("format") != "ai01-swin-mae-training-v1":
            raise ValueError("Resume requires an AI01 SSL training checkpoint")
        # Checkpoints created before --image_size always used 256.
        saved_model_config = {"image_size": 256, **state["model_config"]}
        if saved_model_config != model.training_config() or state["preprocessing"] != preprocessing:
            raise ValueError("Resume model or preprocessing settings do not match")
        if state["dataset_signature"] != signature or state["validation_signature"] != validation_signature:
            raise ValueError("Resume dataset manifests do not match")
        # Permit changing output paths, device and worker count, but not the optimization schedule.
        for name in ("epochs", "batch_size", "accum_iter", "lr", "min_lr", "weight_decay",
                     "warmup_epochs", "clip_grad", "sampling", "no_augment", "array_layout", "max_steps", "seed", "no_amp"):
            if state["args"][name] != getattr(args, name):
                raise ValueError(f"Resume setting differs: {name}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_loss = state["epoch"], state["best_loss"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        logger.info("Resuming after epoch %s", start_epoch)
    (output / "run_config.json").write_text(json.dumps({
        "args": vars(args), "preprocessing": preprocessing,
        "backbone_config": model.config.to_dict(), "ssl_config": model.training_config(),
    }, indent=2), encoding="utf-8")
    (output / "train_files.txt").write_text("\n".join(map(str, dataset.files)) + "\n", encoding="utf-8")
    if validation_dataset:
        (output / "validation_files.txt").write_text(
            "\n".join(map(str, validation_dataset.files)) + "\n", encoding="utf-8"
        )
    for epoch in range(start_epoch, args.epochs):
        logger.info("Starting epoch %s/%s", epoch + 1, args.epochs)
        train_loss = train_epoch(model, loader, optimizer, scaler, device, args, epoch, use_amp,
                                 logger=logger, step_log=step_log)
        validation_loss = validate(model, validation_loader, device, args.seed, use_amp,
                                   epoch=epoch, log_freq=args.log_freq, logger=logger,
                                   step_log=step_log) if validation_loader else None
        if validation_loss is not None and validation_loss < best_loss:
            best_loss = validation_loss
            export_encoder(model.backbone, output / "encoder-best.pt", preprocessing)
            logger.info("Saved encoder-best.pt: validation_loss=%.6f", best_loss)
        stats = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss,
                 "lr": optimizer.param_groups[0]["lr"]}
        logger.info("Epoch summary: %s", json.dumps(stats))
        with (output / "log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stats) + "\n")
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            logger.info("Saving full checkpoint and last encoder for epoch %s...", epoch + 1)
            atomic_save({
                "format": "ai01-swin-mae-training-v1", "model": model.state_dict(),
                "model_config": model.training_config(), "preprocessing": preprocessing,
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch + 1, "best_loss": best_loss, "args": vars(args),
                "dataset_signature": signature, "validation_signature": validation_signature,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "loader_rng": generator.get_state(),
            }, output / "checkpoint-last.pt")
            export_encoder(model.backbone, output / "encoder-last.pt", preprocessing)
            logger.info("Saved checkpoint-last.pt and encoder-last.pt")
    logger.info("Encoder export: %s", output / "encoder-last.pt")


if __name__ == "__main__":
    try:
        main(parser().parse_args())
    except Exception:
        logging.getLogger("ai01_ssl").exception("Training failed")
        raise

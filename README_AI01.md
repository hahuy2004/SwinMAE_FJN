# AI01 — self-supervised Swin-T pretraining

This entry point learns from unlabelled AI01 Z-stacks and exports **the ordinary
Hugging Face `SwinBackbone` used by `A1_Refractor.ipynb`**. There is no custom SSL
module to keep at inference, no RGB conversion and no SwinV2 substitution.

The historical RGB `train.py`, `swin_mae.py`, `swin_unet.py` and `visualize.py`
remain unchanged. **Use `train_ai01.py` for AI01.** The old RGB checkpoints and
visualizer are not interchangeable with this branch.

## Google Colab (clone repository + CLI)

Open [`AI01_SSL_SwinMAE_Colab.ipynb`](AI01_SSL_SwinMAE_Colab.ipynb), enable a GPU,
clone `https://github.com/hahuy2004/SwinMAE_FJN.git` and use
`os.chdir("/content/SwinMAE_FJN")`. The notebook uses the repository's Python
files directly; **no embedded model/trainer code**. Install `requirements-ai01.txt`
and extract `Prepared_Dataset.zip` inside the cloned repository so that
`Prepared_Dataset/train.txt` and `Prepared_Dataset/validation.txt` exist.

The training cell is a direct **`!python -u train_ai01.py ...`** command. Defaults
are one epoch capped at 3 training batches, followed by validation on all 72
held-out samples. For longer training, change the parameters and use a new
`OUTPUT_DIR`. Checkpoints default to a directory inside the clone. Download or
back up files before a runtime reset, or set `OUTPUT_DIR` directly on Drive for
per-epoch persistent saves. Drive backup/download and transfer verification are
optional cells, disabled by default.

For AI01 integration, use `encoder-best.pt`/`encoder-last.pt`, not the full
`checkpoint-last.pt`. Resume examples are provided as Markdown so Run all never
starts a second training job. Drive authentication itself requires Colab.

## 1. What is trained?

```text
unlabelled frame: [11, 400, 400, 1] (or already cropped)
  -> center crop / squeeze / normalize
  -> [B, 11, S, S], S = --image_size (224 or 256; default 256)
  -> 4x4 patch projection: 11 -> 96
  -> mask groups of 4x4 TOKENS (16x16 pixels), default 75%
  -> HF Swin V1 Tiny: depths [2,2,6,2], heads [3,6,12,24], window 7
  -> normalized feature maps at strides 4,8,16,32
  -> hierarchical reconstruction decoder
  -> [B, (S/4)^2, 176] predicted patch pixels
  -> masked-patch MSE against the original normalized stack
```

One sample is **one time point with all 11 Z planes**. Time is not a channel. You
do not need a sequence of 32 frames to pretrain the spatial backbone. Z order is
preserved; augmentations are applied to all planes together. No labels, masks,
PN counts or fake zero images for missing files are generated.

The reconstruction decoder adapts the original repo's patch expansion + Swin
blocks, adding masked multi-scale fusion so that all four backbone output norms
receive gradients. It uses HF Swin blocks/padding too. This is an AI01 adaptation
of Swin MAE, **not a byte-for-byte reproduction of the paper**. Masked token
positions are kept in the dense Swin grid (as in this repo), not removed as in
ViT-MAE; 75% masking does not imply a 75% reduction in encoder memory.

After pretraining, discard the reconstruction decoder and mask token. Transfer
only `model.backbone` into Mask2Former. Mask2Former and the custom BiMamba decoder
are still trained/fine-tuned using labelled AI01 data.

## 2. Environment

From `SSL/Swin-MAE`, in a **separate environment** from the historical RGB repo:

```powershell
python -m pip install -r requirements-ai01.txt
```

Install the appropriate CUDA/CPU PyTorch build for your system first. This branch
uses PyTorch >=2.6 and pins Transformers 4.53.1. Other Transformers versions may
work, but must pass the tests/strict transfer; internal Swin APIs are used for
masked embeddings and decoder blocks. The standalone SSL trainer does not need
Mamba, OpenCV, Albumentations, torchvision or TensorFlow. The AI01 notebook still
requires its existing training dependencies, including `mamba_ssm`.

## 3. Data

For raw `session/DISH/time/*.jpg` images, use the converter at
[`../../data/convert_ai01_raw.py`](../../data/convert_ai01_raw.py). See the
[Vietnamese conversion guide](../../data/README_CONVERT_AI01.md) for verified
sample grouping, square ROI cropping/resizing, previews and train/validation
manifests. Use the generated manifests rather than scanning the entire prepared
root, which contains both splits. The converter additionally requires Pillow.

Supported `.npy` layouts (`--array_layout auto` by default):

| Layout | Shape |
| --- | --- |
| `zhwc` | `[11, H, W, 1]`, original AI01 format |
| `zhw` | `[11, H, W]` |
| `hwz` | `[H, W, 11]` |

Use **raw intensities in [0,255]**, normally uint8, not already normalized arrays
or floats rescaled to [0,1]. Finite values and shapes are checked. Images must be
at least `--image_size` on each spatial axis. With the default 256, a 400x400
input is cropped `[72:328,72:328]`, matching AI01. With 224 it is cropped
`[88:312,88:312]`. This is a **center crop, not a resize**: the 224 setting sees
a smaller field of view at the same pixel scale.
All channels are retained in their stored order, Z0 through Z10.

Normalization matches the notebook's one-channel image processor:

```python
x = (x / 255.0 - mean([0.485, 0.456, 0.406])) / mean([0.229, 0.224, 0.225])
```

Do not point the loader at a directory containing segmentation `.npy` masks or
other arrays. Use a prepared **image-only** root, for example:

```text
prepared_images/
  well_001/
    001.npy
    002.npy
  well_002/
    001.npy
```

Or, preferably, create a UTF-8 text manifest with one image path per line:

```text
# train_unlabelled.txt; relative paths resolve relative to THIS manifest
prepared_images/well_001/001.npy
prepared_images/well_001/002.npy
prepared_images/well_002/001.npy
```

Absolute paths work too. No category folders are required. The trainer never
searches for label metadata or calls the notebook's label-dependent dataset.

**Split by well (or patient/embryo if wells are not independent) BEFORE SSL.**
Exclude supervised validation/test wells from SSL for a conventional held-out
evaluation. Consecutive frames from one well must not cross splits. Optional SSL
validation rejects overlapping parent/well directories; patient-level isolation
still needs to be enforced in your manifests.

Default `--sampling well` samples wells equally in expectation, assuming one
parent directory per well, then frames within that well. This limits domination
by long recordings. Use `--sampling frame` for uniform sampling without
replacement over frames when that grouping is not appropriate. Neither option
guarantees distinct wells or distant time points within every batch.

## 4. Train

Run from the repo directory; replace the example paths with your real data:

```powershell
python train_ai01.py --manifest "F:/data/train_unlabelled.txt" --output_dir "output_ai01_ssl" --epochs 100 --batch_size 4 --accum_iter 4 --lr 0.0001 --device cuda
```

Or scan an image-only root:

```powershell
python train_ai01.py --data_path "F:/data/prepared_images_train" --output_dir "output_ai01_ssl" --epochs 100 --batch_size 4 --device cuda
```

Optional SSL validation from separate wells:

```powershell
python train_ai01.py --manifest "F:/data/train_unlabelled.txt" --val_manifest "F:/data/ssl_val.txt" --output_dir "output_ai01_ssl" --epochs 100 --batch_size 4 --device cuda
```

Start with `--batch_size 1` on a small GPU; use accumulation to increase the
effective batch. CUDA fp16 AMP is on by default; disable with `--no_amp` if
needed. `--device cpu` also works. Workers default to 0 for Windows safety; tune
`--num_workers` for your storage/machine. An epoch draws `len(dataset)` samples;
well-balanced sampling is with replacement.

With fp16 AMP, gradient clipping runs after `GradScaler.unscale_` and lets the
scaler skip updates with Inf/NaN gradients and reduce its scale. Such an overflow
does not itself stop training. Non-finite loss still raises immediately; with
AMP disabled, non-finite gradient norms also remain fatal. A finite loss does
not prove finite gradients, and repeated skipped updates still need diagnosis.

### Select the SSL image size

Use `--image_size 224` or `--image_size 256` (default). This single argument sets
train/validation center crops, encoder resolution, mask grid, reconstruction
shape and checkpoint metadata. The 11 channels and **4x4 patch size remain
unchanged** to match the downstream AI01 Swin-T. No patch-size argument is needed.

The multi-scale decoder is an adaptation, not a proven improvement over a
last-stage-only decoder. Shallow skips may make local reconstruction easier;
they do not copy raw masked pixels (all features come from masked embeddings).
To study mask difficulty without changing the decoder, compare the default
`--mask_group_size 4` (16x16 pixels) with `--mask_group_size 8` (32x32 pixels)
in separate runs, keeping other settings fixed. At size 256, group 8 masks
exactly 75%; at 224 there are 49 groups, so rounding masks 37/49 (about 75.51%).
Adjacent masked groups can form larger regions; a visible token inside every
attention window is not guaranteed. Compare encoders through controlled
downstream fine-tuning, not reconstruction MSE alone. Disabling skips would
require a separate decoder variant and checkpoint configuration; this change
does not silently remove them from existing models.

| Image size | Patch grid | 4x4-token mask groups | Masked patches at 75% |
| --- | --- | --- | --- |
| 224 | 56x56 | 14x14 | 2352 of 3136 |
| 256 | 64x64 | 16x16 | 3072 of 4096 |

For example, in Colab (use a new, empty `OUTPUT_DIR`):

```python
!python -u train_ai01.py \
    --manifest "/content/SwinMAE_FJN/datasets/train.txt" \
    --val_manifest "/content/SwinMAE_FJN/datasets/val.txt" \
    --output_dir "{OUTPUT_DIR}" \
    --image_size 224 \
    --epochs 20 \
    --warmup_epochs 2 \
    --max_steps 0 \
    --batch_size 16 \
    --accum_iter 2 \
    --sampling frame \
    --num_workers 2 \
    --lr 0.0001 \
    --min_lr 0.000001 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --clip_grad 1.0 \
    --seed 42 \
    --log_freq 20 \
    --feature_samples 3 \
    --feature_image_freq 5 \
    --feature_mode both \
    --stage_log_freq 20 \
    --save_freq 1 \
    --device cuda
```

Batch 16 / accumulation 2 keeps the nominal effective batch at 32, like 8 / 4.
It is a configuration to **try**, not a VRAM guarantee. Benchmark a short run
in a separate output directory before committing to a full run on your GPU.
If it runs out of memory, use 8 / 4. A larger microbatch can improve throughput
but does not guarantee better downstream accuracy; 32 / 1 should only be tried
after checking memory and throughput. Do not automatically increase the LR when
the effective batch stays unchanged.

Full checkpoints now store `image_size` in `model_config`, so the usual preview
constructor `AI01SwinMAE(**checkpoint["model_config"])` restores the right size.
Older full checkpoints without that field still default to 256. Resume must keep
the same image size, batch size and accumulation; changing them requires a new
run, optionally initialized with `--init_encoder`.

### Progress and log files

Use `--log_freq 20` (default) or `--log_freq 10` to print train and validation
progress every N **batches**, plus the first and last batch. A step here is a
DataLoader batch, not an optimizer update: with `--accum_iter 4`, weights are
updated after four batches. Logs include current loss, running mean loss,
learning rate (train), elapsed time and estimated remaining time for that phase.

All logs are written incrementally under `--output_dir`:

- `train.log`: timestamped console messages, epoch summaries and CLI exceptions.
- `steps.jsonl`: structured metrics at the configured logging intervals.
- `log.jsonl`: one summary per completed epoch, retained for existing plotting code.

### Stage feature diagnostics

With `--val_manifest`, diagnostics run after the epoch summary and before the
full/last checkpoint save. Defaults are:

- `--feature_samples 3`: up to three fixed validation images, preferring one
  frame from each distinct parent/well directory; fill from remaining frames if
  there are fewer wells. `0` disables all feature logging.
- `--feature_image_freq 5`: PNGs at baseline epoch 0, epoch 1, every fifth epoch, and the final
  epoch. `0` disables PNGs but retains per-epoch statistics.
- `--feature_mode both` (default): logs/saves the masked SSL encoder path and
  the ordinary unmasked backbone forward separately. Either mode can also be
  selected on its own.

Each sample logs all four normalized stage outputs: shape, `finite`, `min`,
`max`, `mean`, `std` and `spatial_std`. `std` is population standard deviation
over the whole feature tensor. `spatial_std` is population standard deviation
over HxW within each channel, averaged over channels. NaN/Inf produces a warning,
a count of non-finite values and JSON `null` statistics, not invalid JSON NaNs.

Additional geometry diagnostics use the full **spatial tokens x channels**
matrix for each sample/stage, centered over tokens for PCA:

- `effective_rank`: exp(entropy of normalized squared singular values), i.e.
  variance-spectrum effective rank. `effective_rank_fraction` divides by
  `max_centered_rank = min(H*W-1, C)`. At stage 4 this maximum is 63 for input
  256 and 48 for input 224, not 768.
- `pca_top1_variance_ratio`, `pca_top3_variance_ratio`: explained-variance
  fractions. Undefined (JSON null) when there is no spatial variance.
- `mean_token_cosine`: mean pairwise cosine of distinct **uncentered**, nonzero
  token vectors, computed without materializing an NxN similarity matrix.
- `low_variance_channel_fraction`: fraction of channels with population spatial
  std below 1e-3. This is a debug threshold, not proof a neuron is dead. Min,
  p10, median and max spatial std, plus the zero-token fraction, are also saved.
- `feature_rel_l2_from_baseline`, `feature_cosine_from_baseline`: change in the
  same fixed sample/mask/stage relative to the saved pre-training reference.

Before a new run trains, the logger records **epoch 0**, labeled `random_init`
or `init_encoder` as appropriate. Baseline features, PCA bases and RGB limits
are persisted in `feature_debug/baseline_<signature>.pt`. Resume reuses these
references (including when the output directory changes but the checkpoint's
feature_debug directory is available), without rewriting epoch-0 images/logs.
An old checkpoint or changed diagnostic sample/mode settings without a matching
reference creates a clearly labeled **resume-epoch baseline**, not fake random
epoch 0. Keep the baseline file with checkpoints to retain comparisons.

Diagnostics run in `eval()` / `inference_mode()` using FP32 and no augmentation.
Images and masks are cached on CPU. Each mask is generated with a dedicated CPU
generator seeded with `--seed + validation_index`, so neither sample selection
nor masks consume the training RNG. They remain fixed between epochs/resumes
with the same data, model settings and seed. The diagnostic pass does not update
weights and restores the previous model training/eval mode.
Ordinary unmasked HF forwards can mutate per-layer window/shift attributes on
small maps; diagnostics also restore those attributes before the next SSL pass.

Example format (numbers below are placeholders, not experimental results):

```text
Epoch summary: {"epoch": 5, ...}
[features] epoch=5 sample=0 mode=masked mask_seed=42 source=.../frame.npy
[features] stage1 shape=(1, 96, 64, 64) finite=True min=... max=... mean=... std=... spatial_std=...
[features] stage2 shape=(1, 192, 32, 32) finite=True min=... max=... mean=... std=... spatial_std=...
[features] stage3 shape=(1, 384, 16, 16) finite=True min=... max=... mean=... std=... spatial_std=...
[features] stage4 shape=(1, 768, 8, 8) finite=True min=... max=... mean=... std=... spatial_std=...
[features] Saved feature_debug/epoch005_val0000_masked.png
Saving full checkpoint and last encoder for epoch 5...
```

The stage shapes above are for 256 input; at 224 they are 56x56, 28x28, 14x14,
7x7. The original channel PNG has four stage rows and four channel columns. The
fixed channel indices are 0, C//3, 2*C//3, C-1. Each panel has its own 1st/99th
percentile contrast limits: colors are not directly comparable amplitudes and
are not segmentation probabilities. Only the fixed reference features are saved
as tensors; no full feature tensors are dumped every epoch. A second
`*_pca.png` shows all four stages as PCA-to-RGB maps. Each sample/mode/stage uses
its own baseline-fitted mean, 3-component basis, and 1st/99th percentile RGB
limits, frozen across epochs. Colors are comparable **over time for that same
panel**, not across different samples or stages, and may saturate as features
drift. PCA structure is not a semantic segmentation prediction.
Plotting uses Matplotlib's headless Agg canvas, installed through the
updated `requirements-ai01.txt`; include the new `ai01_features.py` when copying
files to Colab.

Outputs under `--output_dir`:

- `train.log`: the same feature messages printed to the console.
- `feature_stats.jsonl`: one structured row per epoch/sample/mode, including
  source image, mask seed/hash, actual masked fraction and all stage statistics.
- `feature_debug/samples.json`: the fixed sample indices, paths and mask metadata.
- `feature_debug/epoch005_val0000_masked.png`: example naming for a feature PNG.
- `feature_debug/epoch005_val0000_masked_pca.png`: matching PCA-to-RGB view.

Without validation data, feature logging is explicitly disabled; it does not
silently use augmented training images. Diagnostic options may change on resume
without changing the optimization schedule, but keep them fixed for comparisons.
Mean near zero/std near one can come from LayerNorm, and nonzero spatial
variation is possible even before training. These are debugging signals, not
proof of useful learning. Low spatial rank on one homogeneous image is not
enough to diagnose dataset-wide collapse; compare multiple fixed images/modes.

### Stage learning: gradients and actual parameter changes

`--stage_log_freq 20` (default) logs each stage's gradient L2 norm every 20
**optimizer update attempts**, plus first/last. This is independent of
`--log_freq`, which counts DataLoader batches. `--stage_log_freq 0` disables this
monitor; feature logging remains independently controlled.

Gradients are read **after unscale and before clipping**. Logs distinguish
missing gradients, zero gradients and non-finite gradient norms. AMP skip events
and scale reductions are recorded even between sampled gradient logs. At the
epoch summary, `weight_delta_norm`, `weight_rel_change` (delta norm divided by
epoch-start parameter norm), `changed`, and successful/skipped update counts
show whether parameters really changed. Snapshots are on CPU, once per epoch;
they add roughly one encoder's worth of CPU RAM, not GPU RAM.

Grouping follows HF encoder.layers: each stage includes its layer/downsampling
parameters and corresponding output normalization; stage 1 also includes patch
embeddings. Norms are not normalized for differing parameter counts; do not rank
stage importance by their magnitude. Weight changes may include AdamW decay or
momentum, so use gradient, skip, feature drift and reconstruction loss together.

```text
[learning] epoch=1 update=20/491 stage3 grad_norm=... finite=True grad_none=0 skipped=False
[learning] epoch=1 stage3 weight_rel_change=... changed=True finite=True
[learning] epoch=1 successful_updates=.../491 skipped=...
```

Messages go to console/`train.log`; structured gradient samples, skip events and
epoch parameter deltas go to `stage_learning.jsonl`. This monitor also works
without validation data. Copy the new `ai01_learning.py` along with the trainer.
Neither gradients, parameter changes, non-collapse metrics nor PCA views prove
PN usefulness; downstream fine-tuning/probing is still the deciding evaluation.

`--log_freq` controls logging only. `--save_freq 1` saves full/last checkpoints
after every epoch; the best encoder is saved when validation MSE improves. It
does not save a checkpoint every N logging steps. Logging frequency may be
changed on resume without changing the training schedule.

Manifest paths are relative to the manifest file's own directory. For example,
if a line is `2/well/001.npy` and the image is under `datasets/2/well/001.npy`,
the manifest should be stored in `datasets/`, not copied to its parent directory
without changing its entries. For Colab, keeping images on the local `/content`
disk and saving output checkpoints/logs to mounted Drive avoids many slow image
reads through Drive.

For a **format/training smoke test only**, not a useful pretrained model:

```powershell
python train_ai01.py --manifest "F:/data/train_unlabelled.txt" --output_dir "output_ai01_smoke" --epochs 1 --warmup_epochs 0 --batch_size 1 --max_steps 1 --device cuda
```

The default reconstruction target is the AI01-normalized input. Optional
`--norm_pix_loss` additionally normalizes each target patch; use this as an
ablation, not a silent preprocessing change. With this option, unpatchified
predictions represent patch-normalized targets rather than raw image intensities.

Outputs:

| File | Purpose |
| --- | --- |
| `encoder-last.pt` | **Encoder-only export to load in AI01** |
| `encoder-best.pt` | Best encoder by SSL validation MSE; only when validation is supplied |
| `checkpoint-last.pt` | Full SSL model/decoder/optimizer/scaler/RNG state for resume |
| `run_config.json` | Architecture, preprocessing and training arguments |
| `train_files.txt`, `validation_files.txt` | Actual data manifests (validation file is optional) |
| `log.jsonl` | Training/validation MSE and learning rate |
| `train.log` | Timestamped console progress, summaries and errors |
| `steps.jsonl` | Train/validation batch metrics at `--log_freq` intervals |
| `feature_stats.jsonl` | Per-epoch fixed-sample stage activation statistics |
| `stage_learning.jsonl` | Unscaled/pre-clip gradient norms, AMP skips and per-epoch weight deltas |
| `feature_debug/` | Sample metadata and periodic four-stage PNG grids |

Encoder exports contain all state-dict tensors, backbone config, preprocessing
metadata and Transformers version. They do **not** include the SSL decoder.
`checkpoint-last.pt` is intentionally a different format and is rejected by the
AI01 encoder loader. Files are PyTorch dictionaries, not RGB Hub checkpoints.

Resume from an epoch boundary using the same training arguments, plus:

```powershell
python train_ai01.py --manifest "F:/data/train_unlabelled.txt" --output_dir "output_ai01_ssl" --epochs 100 --batch_size 4 --accum_iter 4 --lr 0.0001 --device cuda --resume "output_ai01_ssl/checkpoint-last.pt"
```

Resume checks manifests and optimization/model settings; don't change the
schedule silently. A new SSL run with a different schedule can use
`--init_encoder "output_ai01_ssl/encoder-last.pt"` instead (new decoder and
optimizer). Only load checkpoints from trusted sources.

## 5. Integrate into `A1_Refractor.ipynb`

The notebook now has two optional settings in its configuration cell:

```python
SSL_ENCODER_CHECKPOINT = "F:/checkpoints/ai01_ssl/encoder-last.pt"
SSL_CODE_DIR = Path("F:/WFH_APP/Fujinet/AI01/SSL/Swin-MAE")
```

Set paths for the machine actually running the notebook. On Linux, use Linux
paths. Leave `SSL_ENCODER_CHECKPOINT = ""` to preserve random initialization.

The model-initialization cell now:

1. Sets backbone metadata to 11 channels and image size 256.
2. Creates the model and 11-channel patch projection.
3. Loads SSL encoder weights **strictly**, before Trainer/optimizer creation.
4. Keeps the existing custom Mask2Former forward and BiMamba decoder replacement.

The loader checks architecture, preprocessing, every tensor key and every shape
before copying weights. SSL crop size is allowed to differ: an encoder pretrained
at 224 can be loaded into AI01 at 256 without changing weights because this Swin
uses relative window positions, not absolute image-position embeddings. All
other preprocessing fields (including Z order and normalization) must still
match. This does not change AI01's input crop to 224 or retain the SSL decoder.
**No `strict=False`, discarded layers or silent partial
load.** Do not recreate the patch projection AFTER loading SSL. Do not later
load an old full-model checkpoint or resume an unrelated supervised run unless
you intend to overwrite the SSL initialization.

Without notebook integration, equivalent Python is:

```python
from ai01_transfer import load_ai01_encoder

# hf_model is already a Mask2Former with an 11-channel Swin-T backbone.
report = load_ai01_encoder(hf_model, "encoder-last.pt", image_processor=image_processor)
print(report)
```

To use the exported model as a standalone backbone:

```python
from ai01_transfer import load_exported_backbone

backbone = load_exported_backbone("encoder-last.pt").eval()
# Input must use the saved AI01 preprocessing, shape [N,11,256,256].
features = backbone(pixel_values).feature_maps
# [N,96,64,64], [N,192,32,32], [N,384,16,16], [N,768,8,8]
```

## 6. Verify compatibility

Offline automated tests (synthetic data; no Hub/network needed):

```powershell
python -m unittest discover -s tests -v
```

Tests cover processor/crop/layout parity with the notebook, 11-channel patch
round-trip, grouped masking, masked-only loss, no masked-pixel leakage, gradients
through all encoder/decoder parameters, exact feature equality after strict
transfer, rejected incompatible exports, and a full ordinary Mask2Former forward
at 256x256. Notebook code-cell syntax and SSL loading order are also checked.

For your actual export and the **actual AI01 Hub configuration**:

```powershell
python verify_ai01.py --encoder "output_ai01_ssl/encoder-last.pt" --device cuda
```

This downloads only the configuration of
`facebook/mask2former-swin-tiny-coco-instance`, not its pretrained weights. For
offline use, pass `--mask2former_config "path/to/local/config_directory"`.
It checks exact feature equality after transfer and executes a Mask2Former
forward. It does not execute the notebook's custom Mamba decoder or its dataset.

### Checks performed during this implementation

On PyTorch 2.7.0 / Transformers 4.53.1, the offline suite passed. A separate
CUDA fp16 smoke run on synthetic 11x400x400x1 data also completed one training
step, SSL validation, encoder/full-checkpoint save and resume loading. The
export loaded all **237 backbone tensors** against the actual Hub Mask2Former
configuration, produced exactly matching backbone features, and returned class
logits `[1,7,3]` and mask logits `[1,7,64,64]`. These are compatibility checks,
not evidence of pretraining quality or an execution of the full BiMamba notebook.

**Compatibility is testable; improved accuracy is not guaranteed.** The backbone
is Swin Transformer itself, not a substitute merely producing similar shapes.
Its architecture/weights interface can be verified. You still need real SSL
training and controlled supervised fine-tuning experiments to measure benefit.
SSL MSE alone does not measure PN segmentation quality. Compare random vs SSL
initialization with identical labelled splits, settings and evaluation metrics.

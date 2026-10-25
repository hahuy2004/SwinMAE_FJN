# AI01 — self-supervised Swin-T pretraining

This entry point learns from unlabelled AI01 Z-stacks and exports **the ordinary
Hugging Face `SwinBackbone` used by `A1_Refractor.ipynb`**. There is no custom SSL
module to keep at inference, no RGB conversion and no SwinV2 substitution.

The historical RGB `train.py`, `swin_mae.py`, `swin_unet.py` and `visualize.py`
remain unchanged. **Use `train_ai01.py` for AI01.** The old RGB checkpoints and
visualizer are not interchangeable with this branch.

## 1. What is trained?

```text
unlabelled frame: [11, 400, 400, 1] (or already cropped)
  -> center crop / squeeze / normalize
  -> [B, 11, 256, 256]
  -> 4x4 patch projection: 11 -> 96
  -> mask groups of 4x4 TOKENS (16x16 pixels), default 75%
  -> HF Swin V1 Tiny: depths [2,2,6,2], heads [3,6,12,24], window 7
  -> normalized feature maps at strides 4,8,16,32
  -> hierarchical reconstruction decoder
  -> [B, 4096, 176] predicted patch pixels
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
at least 256x256. A 400x400 input is cropped `[72:328,72:328]`, matching AI01.
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
before copying weights. **No `strict=False`, discarded layers or silent partial
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

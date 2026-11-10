# AI01 — Swin V2 Tiny MAE

Phiên bản **riêng cho Swin Transformer V2**, chạy trong `SSL/Swin-MAE`.
Không thay thế model/checkpoint Swin V1 và không cần download trọng số RGB.
Chạy bằng `python -m v2.ai01_train_v2` từ root repo, không chạy trực tiếp file trong `v2/`.
Notebook: [`ai01_colab_v2.ipynb`](ai01_colab_v2.ipynb).

## 1. Tham chiếu và phạm vi

- `../../../2/2111.09883v2.pdf`: mục 3.2 (residual-post-norm, cosine attention),
  3.3 (Log-CPB), 3.6 (cấu hình Tiny).
- `../../../2/Swin-Transformer/models/swin_transformer_v2.py`: implementation
  **chính thức Microsoft**, làm chuẩn kiến trúc; đặc biệt `WindowAttention`,
  `SwinTransformerBlock`, `PatchMerging`, `_init_respostnorm` và quy tắc no-decay.
- `../../../2/Swin-Transformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml`:
  Tiny, window 8, ảnh 256, drop-path 0.2.
- `../../../2/Swin-Transformer-V2/swin_transformer_v2/model_parts.py`: tham khảo
  bản **không chính thức**. Không copy nguyên vì có khác biệt với Microsoft,
  ví dụ PatchMerging vẫn normalize 4C trước reduction.
- Runtime dùng `transformers==4.53.1`, **`Swinv2Backbone` / `Swinv2Stage`**.
  Không import hai repository tham chiếu lúc train; không cần `timm`.

Đây là **Swin-MAE adaptation cho AI01 với encoder/decoder Swin V2**, không phải
tái hiện nguyên SimMIM của bài Swin V2, không phải sparse ViT-MAE, và không phải
kiến trúc SSL chính thức mới của Microsoft. Mục tiêu downstream là kiểm chứng
bằng fine-tuning/segmentation trên dữ liệu có nhãn, không chỉ nhìn ảnh tái tạo.

## 2. Kiến trúc

```text
Một time point chứa đủ 11 Z
  -> center crop + normalization giống nhánh AI01 V1
  -> [B, 11, 256, 256]
  -> patch projection 4x4, 11 -> 96
  -> che cùng vị trí trên toàn bộ 11 Z bằng learnable mask token
     nhóm mask: 4x4 tokens = 16x16 pixels; mặc định 75%
  -> HF Swin V2 Tiny
     depths [2,2,6,2], heads [3,6,12,24], window 8
     cosine attention + learned logit_scale + Log-CPB + residual-post-norm
     PatchMerging: concat 4C -> Linear 2C -> LayerNorm 2C
  -> stage maps:
     [B, 96,64,64], [B,192,32,32], [B,384,16,16], [B,768,8,8]
  -> hierarchical reconstruction decoder:
     patch expansion + masked multiscale fusion + Swin V2 blocks
  -> [B,4096,176] = mỗi patch 4x4x11 giá trị
  -> chọn Z05/Z06/Z07 -> masked-patch MSE
```

Các residual LayerNorm trong block được khởi tạo **weight=0, bias=0**, theo
`_init_respostnorm` của Microsoft. Do đó ở bước đầu gradient của một số trọng số
attention/MLP bằng 0 là có chủ đích; norm học trước rồi mở đường gradient.
`logit_scale`, CPB MLP, bias/norm và mask token không chịu weight decay.

Decoder chỉ dùng lúc SSL. Khi inference/downstream, chỉ giữ backbone V2 thường;
không mang theo mask token, reconstruction decoder hay hàm forward SSL.

### Dữ liệu và mục tiêu

- Dùng chung `utils/ai01_data.py`, giữ thứ tự Z0..Z10, crop và normalization của V1.
- Nhận `.npy` `[11,H,W,1]`, `[11,H,W]`, `[H,W,11]` qua `--array_layout`.
- Intensities thô `[0,255]`; không chuẩn hóa trước lần nữa.
- Mặc định input **đủ 11 Z**, chỉ loss/metrics trên `--target_z 5 6 7`.
  Đây là **zero-based**, tương ứng mặt phẳng thứ 6, 7, 8 trong mảng.
- Tám Z còn lại là context, không phải reconstruction output đã được giám sát.
- Cùng spatial mask che tất cả Z, không để lộ target qua các kênh chưa mask.
- Không cần nhãn/mask PN. Chia train/validation/test theo well/patient trước SSL;
  không để frame cùng well hoặc cùng patient lọt qua các split.
- Mask giữ nguyên dense grid; 75% masking **không** giảm 75% bộ nhớ encoder.

## 3. Các file

| File | Chức năng |
|---|---|
| `v2/ai01_model_v2.py` | Cấu hình Tiny, encoder, decoder V2, masking và loss |
| `v2/ai01_train_v2.py` | CLI train/resume/init encoder |
| `v2/ai01_learning_v2.py` | Optimizer grouping và gradient/update của bốn stage |
| `v2/ai01_transfer_v2.py` | Export/load strict, tạo Mask2Former V2 với task head mới |
| `v2/ai01_evaluate_v2.py` | Reconstruction evaluation từ full checkpoint |
| `v2/ai01_verify_v2.py` | Kiểm tra encoder export và tùy chọn Mask2Former forward |
| `utils/ai01_requirements.txt` | Môi trường SSL dùng chung |
| `utils/ai01_requirements_downstream.txt` | Thêm SciPy để chạy Mask2Former/tests |
| `tests/test_ai01_v2.py` | Kiểm thử offline, không tải Hub/dataset |

V1/V2 dùng chung `utils/ai01_train.py`, `utils/ai01_evaluate.py`, data, targets,
metrics, feature diagnostics và các lớp trung lập kiến trúc. `v2/` không import
`v1/`; `utils/` không import hai package phiên bản. Không monkey-patch global model.

## 4. Cài đặt và train

Chạy từ `SSL/Swin-MAE`. Cài PyTorch phù hợp CUDA của máy trước, rồi:

```powershell
python -m pip install -r utils/ai01_requirements.txt
```

Smoke test với **dữ liệu thật của bạn**, thay đường dẫn manifest:

```powershell
python -u -m v2.ai01_train_v2 --manifest "Prepared_Dataset/train.txt" --val_manifest "Prepared_Dataset/validation.txt" --output_dir "output_ai01_v2_smoke" --epochs 1 --warmup_epochs 0 --batch_size 1 --max_steps 3 --feature_samples 0 --device cuda
```

Train đầy đủ:

```powershell
python -u -m v2.ai01_train_v2 --manifest "Prepared_Dataset/train.txt" --val_manifest "Prepared_Dataset/validation.txt" --output_dir "output_ai01_v2_ssl" --epochs 100 --image_size 256 --window_size 8 --batch_size 2 --accum_iter 4 --lr 0.0001 --warmup_epochs 10 --target_z 5 6 7 --gradient_checkpointing --device cuda
```

- Không có GPU: `--device cpu --no_amp`; nên smoke ít batch trước.
- `--max_steps` giới hạn **training batches mỗi epoch**, không giới hạn validation.
- CLI hỗ trợ `--image_size 224` hoặc `256`; V2 sẽ pad window khi cần và tắt shift
  ở stage cuối nếu feature map không lớn hơn window. 256/window8 là mặc định chuẩn.
- `--feature_samples 0` tắt feature diagnostics; `--feature_image_freq 0` chỉ ghi
  thống kê, không vẽ PNG. `--stage_log_freq 0` tắt gradient diagnostics.
- `--norm_pix_loss` chuẩn hóa target theo patch, chỉ dùng target Z; khi bật,
  intensity reconstruction metrics tự tắt vì không còn cùng thang đo.
- Muốn all-Z ablation: `--target_z 0 1 2 3 4 5 6 7 8 9 10`.
- Không ghi đè run cũ: dùng output folder mới, hoặc `--resume`.

### Checkpoint và resume

```text
checkpoint-last.pt  Full MAE + optimizer/scaler/RNG; dùng resume/evaluation
encoder-last.pt     Chỉ V2 encoder; dùng chuyển sang downstream
encoder-best.pt     Chỉ V2 encoder với SSL validation loss thấp nhất (nếu có val)
```

Resume giữ **nguyên toàn bộ tham số run cũ**, chỉ thêm:

```text
--resume "output_ai01_v2_ssl/checkpoint-last.pt"
```

Resume ở ranh giới epoch, không phải giữa batch. Đổi target Z, kiến trúc, masking
hoặc optimization schedule sẽ bị từ chối. Nếu đổi objective, tạo output mới và
dùng `--init_encoder "old_v2_run/encoder-best.pt"` để khởi tạo encoder, decoder và
optimizer mới. `--init_encoder` chỉ nhận V2 export tương thích, không nhận V1/RGB.

## 5. Evaluation

```powershell
python -m v2.ai01_evaluate_v2 --checkpoint "output_ai01_v2_ssl/checkpoint-last.pt" --val_manifest "Prepared_Dataset/validation.txt" --output_dir "output_ai01_v2_eval" --batch_size 2 --mask_seeds 42 43 44 --device cuda
```

Ghi masked PSNR/SSIM/MAE/RMSE/gradient MAE trên target Z, frame/well macro,
cùng `steps.jsonl`, `train.log` và báo cáo JSON/CSV. Evaluation output phải là
folder trống. Không thể tái tạo ảnh chỉ từ encoder export vì thiếu decoder SSL.

## 6. Áp dụng vào AI01 / Mask2Former

**Không nạp vào Swin V1 hiện tại rồi dùng `strict=False`.** Cần tạo downstream với
`Swinv2Config` giống export. Với Tiny, số kênh/stride vẫn `[96,192,384,768]` /
`[4,8,16,32]`, thuận tiện giữ pixel decoder và các temporal/BiMamba head hiện có.

Ví dụ standalone:

```python
from v2.ai01_transfer_v2 import load_exported_backbone

encoder = load_exported_backbone("output_ai01_v2_ssl/encoder-best.pt").eval()
# x phải là tensor đã crop/normalize đúng: [B,11,256,256]
features = encoder(x).feature_maps
```

Ví dụ Mask2Former **task head mới, cần supervised fine-tuning**:

```python
from v2.ai01_transfer_v2 import build_ai01_v2_mask2former

model = build_ai01_v2_mask2former(
    "output_ai01_v2_ssl/encoder-best.pt",
    num_labels=2,  # thay bằng cấu hình class thật của AI01
    num_queries=100,
)
# Tạo optimizer SAU khi nạp encoder; train/fine-tune bằng nhãn AI01.
```

Nếu tự dựng AI01 có custom heads, dùng `read_export(path)["backbone_config"]`
để dựng `Mask2FormerConfig`, rồi sau khi hoàn tất model:

```python
from v2.ai01_transfer_v2 import load_ai01_v2_encoder

report = load_ai01_v2_encoder(model, "output_ai01_v2_ssl/encoder-best.pt", image_processor)
```

Hàm kiểm tra loại V2, full config, preprocessing, keys và shape **trước khi copy**;
không sửa decoder/head. Helper không tự xây lại các custom temporal/BiMamba module
trong notebook AI01. Các notebook downstream chỉ được cập nhật đường dẫn import;
không tự động chuyển kiến trúc V1 đang dùng sang V2.

V2 resolve window/shift khi khởi tạo nên strict transfer kiểm tra cả `image_size`
và `pretrained_window_sizes`. Chuyển ảnh/window resolution khác cần quy trình
rebuild/transfer Log-CPB riêng; **không tự động hỗ trợ** bằng cách bỏ kiểm tra.
HF Mask2Former 4.53.1 có thể cảnh báo model type `swinv2` chưa nằm trong whitelist
(`swin`), dù AutoBackbone xây được V2. Vì vậy cần chạy verification và kiểm thử
fine-tuning thực tế, không suy luận chỉ từ việc khởi tạo thành công.

```powershell
python -m pip install -r utils/ai01_requirements_downstream.txt
python -m v2.ai01_verify_v2 --encoder "output_ai01_v2_ssl/encoder-best.pt" --mask2former --device cpu
python -m unittest discover -s tests -p "test_ai01_v2.py" -v
```

Test dùng cả Tiny chuẩn (forward/backward) và model V2 thu nhỏ để train/resume/eval
nhanh trên CPU. Dữ liệu ngẫu nhiên trong test chỉ để kiểm tra code, **không phải**
kết quả SSL/segmentation trên AI01. CUDA AMP và chất lượng downstream cần kiểm tra
trên môi trường GPU/dataset thực tế.

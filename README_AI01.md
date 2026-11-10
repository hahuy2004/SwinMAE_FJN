# AI01 Swin-MAE — V1 / V2 / shared utils

## Cấu trúc và quy tắc tên

File riêng phiên bản: **`ai01_<chức năng>_v1.py` / `ai01_<chức năng>_v2.py`**.
File dùng chung: **`utils/ai01_<chức năng>.py`**, không nhân bản theo phiên bản.

```text
Swin-MAE/
├── swin_mae.py                 # Code RGB gốc, giữ nguyên vị trí
├── swin_unet.py
├── train.py
├── visualize.py
├── requirements.txt            # Môi trường RGB gốc, tách biệt AI01
├── v1/
│   ├── ai01_model_v1.py
│   ├── ai01_train_v1.py
│   ├── ai01_evaluate_v1.py
│   ├── ai01_transfer_v1.py
│   ├── ai01_learning_v1.py
│   ├── ai01_verify_v1.py
│   ├── ai01_colab_v1.ipynb
│   └── README.md
├── v2/
│   ├── ai01_model_v2.py
│   ├── ai01_train_v2.py
│   ├── ai01_evaluate_v2.py
│   ├── ai01_transfer_v2.py
│   ├── ai01_learning_v2.py
│   ├── ai01_verify_v2.py
│   ├── ai01_colab_v2.ipynb
│   └── README.md
├── utils/
│   ├── ai01_data.py            # Đọc NPY, crop, normalize, chia well
│   ├── ai01_targets.py         # Chọn Z giám sát; chung cho cả V1/V2
│   ├── ai01_metrics.py         # Masked PSNR/SSIM/MAE/RMSE/gradient MAE
│   ├── ai01_features.py        # Thống kê/ảnh feature và baseline
│   ├── ai01_layers.py          # PatchExpansion trung lập kiến trúc
│   ├── ai01_preprocessing.py   # Kiểm tra preprocessing khi transfer
│   ├── ai01_learning.py        # Vòng ghi gradient/update diagnostics
│   ├── ai01_train.py           # Vòng train/resume dùng chung
│   ├── ai01_evaluate.py        # Vòng evaluation dùng chung
│   ├── ai01_requirements.txt
│   ├── ai01_requirements_downstream.txt
│   └── ...                    # Các utility RGB gốc được giữ lại
└── tests/
```

`v1/` và `v2/` chỉ import `utils/` và thư viện ngoài, không import lẫn nhau.
`utils/` không import model/package phiên bản; entry point truyền model và hooks.
Hai model vẫn khác attention/norm/position bias, không chia sẻ checkpoint cho nhau.
Các file Swin-MAE RGB gốc và utility RGB vẫn ở vị trí cũ. Chỉ code AI01 được
tổ chức trong `v1/`, `v2/`, `utils/`; không dùng `train.py` RGB cho dữ liệu 11 Z.

## Chạy từ root `SSL/Swin-MAE`

Cài PyTorch phù hợp CUDA/CPU trước. V1 và V2 dùng chung môi trường:

```powershell
python -m pip install -r utils/ai01_requirements.txt
# Thêm SciPy nếu cần Mask2Former hoặc full tests:
python -m pip install -r utils/ai01_requirements_downstream.txt
```

```powershell
python -m v1.ai01_train_v1 --manifest "Prepared_Dataset/train.txt" --val_manifest "Prepared_Dataset/validation.txt" --output_dir "output_ai01_v1_ssl" --device cuda
python -m v2.ai01_train_v2 --manifest "Prepared_Dataset/train.txt" --val_manifest "Prepared_Dataset/validation.txt" --output_dir "output_ai01_v2_ssl" --gradient_checkpointing --device cuda
```

Dùng `python -m ...`, **không** `python v1/ai01_train_v1.py`, không `cd v1`/`cd v2`.
Manifest, output và checkpoint tiếp tục resolve từ working directory; file NPY
trong manifest vẫn resolve từ thư mục chứa manifest, không từ package code.

| Chức năng | V1 | V2 |
|---|---|---|
| Train | `python -m v1.ai01_train_v1` | `python -m v2.ai01_train_v2` |
| Evaluate | `python -m v1.ai01_evaluate_v1` | `python -m v2.ai01_evaluate_v2` |
| Verify | `python -m v1.ai01_verify_v1` | `python -m v2.ai01_verify_v2` |
| Colab | [v1/ai01_colab_v1.ipynb](v1/ai01_colab_v1.ipynb) | [v2/ai01_colab_v2.ipynb](v2/ai01_colab_v2.ipynb) |
| Chi tiết | [v1/README.md](v1/README.md) | [v2/README.md](v2/README.md) |

## Đổi từ đường dẫn cũ

Không giữ các wrapper cũ ở root để tránh hai tên cho cùng một chức năng:

| Cũ | Mới |
|---|---|
| `train_ai01.py` | `python -m v1.ai01_train_v1` |
| `ai01_v2_train.py` | `python -m v2.ai01_train_v2` |
| `evaluate_ai01.py` / `ai01_v2_evaluate.py` | `v1/ai01_evaluate_v1.py` / `v2/ai01_evaluate_v2.py` |
| `verify_ai01.py` / `ai01_v2_verify.py` | `v1/ai01_verify_v1.py` / `v2/ai01_verify_v2.py` |
| `ai01_swin_mae.py` / `ai01_v2_swin_mae.py` | `v1/ai01_model_v1.py` / `v2/ai01_model_v2.py` |
| `ai01_transfer.py` / `ai01_v2_transfer.py` | `v1/ai01_transfer_v1.py` / `v2/ai01_transfer_v2.py` |
| `ai01_data.py`, `ai01_targets.py`, `ai01_metrics.py`, `ai01_features.py` | Cùng tên, nằm trong `utils/` |
| `requirements-ai01.txt`, `ai01_v2_requirements.txt` | `utils/ai01_requirements.txt` |
| `ai01_v2_requirements_downstream.txt` | `utils/ai01_requirements_downstream.txt` |
| `AI01_SSL_SwinMAE_Colab.ipynb` | `v1/ai01_colab_v1.ipynb` |

Import vào notebook AI01 (thêm **root Swin-MAE**, không phải thư mục v1/v2, vào sys.path):

```python
from v1.ai01_transfer_v1 import load_ai01_encoder
from v2.ai01_transfer_v2 import load_ai01_v2_encoder
from utils.ai01_data import DEFAULT_PREPROCESSING
from utils.ai01_targets import validate_target_z
```

`checkpoint_model_config` dành riêng cho định dạng checkpoint nằm trong model
từng phiên bản, không nằm trong shared targets:

```python
from v1.ai01_model_v1 import checkpoint_model_config  # Legacy all-Z fallback của V1
# hoặc: from v2.ai01_model_v2 import checkpoint_model_config
```

**Không thay đổi state-dict keys, format metadata, target Z, loss, lịch train hoặc
đường dẫn dataset/output hiện có.** Encoder/full checkpoint cũ vẫn dùng với đúng
phiên bản, không cần train lại chỉ vì đổi layout. Resume vẫn cần cùng tham số run.
Không sửa/xóa checkpoint hoặc dữ liệu có sẵn.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Các bài test kiểm tra V1/V2, train/resume/evaluate, strict transfer, Mask2Former,
import isolation, module CLI và notebook/path migration. GPU/Colab/Drive thực tế
vẫn cần môi trường tương ứng để kiểm chứng, không suy ra chất lượng SSL từ test.

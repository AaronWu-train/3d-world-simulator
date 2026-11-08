# Amber 摺布資料轉換

這是一份 **以單眼估測深度建立的實驗資料集**，可接本專案的點雲 VAE 與 latent dynamics 訓練流程。它不是真實 RGB-D，也不是完整的布料物理模型。原始資料保留在 `amber_data/`，轉換結果在 `data/amber_cloth_estimated/`。

## 目前輸出

來源：`amber_data/20261017_152333_cloth_folding_split/{folding,unfolding}`。

| 分類 | 訓練 episodes | 驗證 episodes |
|---|---:|---:|
| folding | 40 | 4 |
| unfolding | 44 | 5 |
| 合計 | 84 | 9 |

每段保留 15 秒，從 20 FPS 每 4 幀取 1 幀，輸出 **5 FPS、320×240、每段 75 幀**。共 6,975 幀。`seed=42` 在各分類內按整段示範切分，原始 episode 不跨訓練與驗證集。分類依你提供的 splitting 與編號對照保留，沒有重新判定動作成功與否。

```text
data/amber_cloth_estimated/
  train/episode_<原始編號>.hdf5
  val/episode_<原始編號>.hdf5
  conversion.json       # 模型版本、尺度假設、校正來源及 hash、episode 對照
  validation.json       # 資料與讀取器驗證結果
  preview.png           # RGB／估測深度／點雲預覽
  preview.ply           # 可用點雲工具開啟的彩色點雲
```

影像透過影片 PTS 配對 Parquet timestamp；檢查半個來源影格內的誤差，對不上就停止。動作和實際狀態各自保留，不混用。

## 深度與相機

- 深度：[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)，固定 revision `5426e4f0f36572d16453bbda7a8389317b1bef99`。這是相對深度模型，**不是公尺量測器**。
- 本次 CPU 實驗使用 inference size 196，再上採樣至輸出尺寸。可在另一次轉換指定 `--inference-size 518` 取得較細的估測，耗時會增加。
- 使用訓練集參考影像的預測分位數，擬合一組固定的 inverse-depth 映射到假設的 0.4–1.4 公尺。每幀共用同一映射，不逐幀重設尺度，也不使用驗證集擬合尺度。範圍與映射記在 `conversion.json`；此尺度仍需實際深度才能校正。
- 外參來源：`/tmp2/amberho/piper_lerobot/data/cam_in_base.txt`，已複製至 `amber_data/piper_lerobot_assets/calibration/cam_in_base.txt`。檔案明確標示 camera-to-base，因此 `E_cam2world=T_cam_in_base`，`viewmat=inverse(T_cam_in_base)`，世界座標是 Piper 底座。
- 內參來源：`/tmp2/amberho/piper_lerobot/real2sim/episode_000/calibration/intrinsics.json`，已複製至相同的 `calibration/` 目錄。原始解析度 640×480，矩陣依輸出尺寸縮放；320×240 的 `fx=fy≈279.3076, cx=160, cy=120`。檔案指定零畸變。
- 校正與錄製相機的對應依使用者提供的資訊採用，資料中沒有相機序號可獨立核實。外參檔記錄手眼平移殘差 mean 1.9 cm、max 5.23 cm，並非無誤差真值。

只有單一相機看到的表面，遮擋、物體背面與接觸幾何仍不可得。單眼估測也可能隨時間抖動。即使套用外參，點雲尺寸與桌面高度仍可能偏離真實位置。

## 動作與訓練改動

`action/joint_action` 保存原始 7 維絕對目標：六個關節角度（度）＋夾爪寬度（毫米）。`observation/joint_state` 保存實際回讀值。單位沿用錄製專案的預設設定，沒有轉成末端位姿。

為相容現有 latent exporter，`action/cartesian_action` 是 `joint_action` 的 HDF5 hard link；這只是歷史欄位名稱，**內容不是 Cartesian action**，欄位及 episode 屬性都寫明語意。動態模型原本就依資料讀取 action dimension，因此可使用 7 維。來源 frame index、timestamp、video timestamp、task、原始 episode 編號也保留在 HDF5。

請使用新的 `--task amber_cloth_estimated`，它設定：

- 無觸覺、無腕部相機。
- 不讀取不存在的 robot label cache，不對缺少的機器人／布料標籤計算語意損失。
- 粒子重建走通用場景分支；不把這些點宣稱為已分類的布料或機器人。
- 不以估測深度進行立體深度損失或自由空間裁切，也不假造 disparity、baseline 或 gripper masks。
- 不初始化依賴實測背景的 Gaussian bank，保留可見場景的點雲與 RGB 重建訓練。

原有任務預設仍使用原本的監督流程。

## 使用已轉好的資料

從專案根目錄執行。訓練需要已安裝本專案相依套件、驅動正常的 NVIDIA GPU；本次環境沒有可用 CUDA，因此沒有實際啟動 GPU 訓練。

```bash
conda activate 3dws

# 先跑少量訓練／驗證 iteration，確認你的 GPU 環境
python scripts/train_vae.py --task amber_cloth_estimated \
  --data-root data --out-dir runs/amber/vae_smoke \
  --n-epochs 1 --iters 1 --val-iters 1 --num-workers 0

# 正式訓練 VAE
python scripts/train_vae.py --task amber_cloth_estimated \
  --data-root data --out-dir runs/amber/vae

# 訓練完成後匯出 latents
python scripts/export_latents.py \
  --vae-ckpt runs/amber/vae/checkpoints/latest.pth \
  --packed-dir data/amber_cloth_estimated/train --out-dir runs/amber/latents/train
python scripts/export_latents.py \
  --vae-ckpt runs/amber/vae/checkpoints/latest.pth \
  --packed-dir data/amber_cloth_estimated/val --out-dir runs/amber/latents/val

# 接著訓練動態模型；沒有物體標籤，不加物體權重
python scripts/train_dynamics.py \
  --latent-dir runs/amber/latents/train --val-latent-dir runs/amber/latents/val \
  --out-dir runs/amber/dynamics --n-history 8 --object-alpha 1
```

輸出間隔為 0.2 秒，所以 `--n-history 8` 涵蓋約 1.6 秒，與原始 20 FPS 的時間範圍不同。folding/unfolding 的 task 屬性供追蹤來源，目前 dynamics 沒有文字任務條件輸入，學習的是觀測與動作對未來狀態的影響。

## 檢查訓練好的 VAE 重建

在可見 GPU 的 `3dws` 環境執行。若使用 Conda 的 CUDA toolkit，先補齊標頭搜尋路徑，避免 gsplat 編譯找不到 `cuda_runtime.h`：

```bash
conda activate 3dws
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"

# 驗證集：episode 21 是 folding，32 是 unfolding
# 預設各重建第一張、中間、最後一張
python amber_data/script/reconstruct_vae.py --episodes 21 32

# 重建兩段完整示範，另輸出 H.264 對照影片（需要 ffmpeg）
python amber_data/script/reconstruct_vae.py --episodes 21 32 --video

# 完整驗證集：9 段、675 張影格
python amber_data/script/reconstruct_vae.py --all-episodes --video \
  --out-dir runs/amber/reconstruction_all_val
```

預設讀取 `runs/amber/vae/checkpoints/best.pth`、`data/amber_cloth_estimated/val`，輸出至 `runs/amber/reconstruction/`。可用 `--vae-ckpt`、`--packed-dir`、`--out-dir` 更換路徑；`--frames 0 20 40` 可指定影格。

- `overview.png`：各影格的原始 RGB／VAE 重建／RGB 絕對誤差。誤差圖固定使用 0–1 尺度，沒有逐張拉高對比。
- `episode_*_frame_*.png`：各影格的對照圖。
- `episode_*_frame_*.ply`：重建 Gaussian 中心的彩色點雲（最多約 50,000 點），供一般點雲工具檢視；不包含完整 Gaussian 橢球參數。
- `episode_*_comparison.mp4`：使用 `--video` 時輸出的完整對照影片。
- `report.json`：checkpoint hash、訓練 epoch、所選影格，以及 RGB MAE、PSNR 與透明度摘要。影片模式計算所有影格，快照模式只計算所選影格。
- `evaluation.md`：逐段平均誤差、最低 PSNR 及對應時間，連結每段影片和較差影格。
- `episode_*_worst.png`、`worst_frames_overview.png`：各段 PSNR 最低的影格。這是按影像誤差選出，不代表有真實的抓取／接觸標籤。

工具使用已訓練的編碼器、解碼器和粒子 head；不更新權重，解碼時不提供真實占用網格，也不提供物體標籤。這是在重建**已看到的當前影格**，未使用動作預測未來。

MAE 與 PSNR 是整張 RGB 畫面的誤差，會受大量桌面背景影響。請同時看布料輪廓、摺痕、夾爪邊緣和整段影片的穩定性；原視角外觀接近並不能證明其他視角的 3D 幾何正確。

## 重跑轉換與驗證

轉換可在 CPU 上執行。額外相依套件列在 `amber_data/script/requirements-convert.txt`。本次使用既有的 `/tmp2/amberho/piper_lerobot/.venv/bin/python`，沒有修改該環境。以下命令都從專案根目錄執行。

模型檔已下載到專案的 `.cache/depth-anything-v2-small/`。若在別台機器執行，先取得上述固定版本的 `config.json`、`preprocessor_config.json`、`model.safetensors` 放到相同目錄。

```bash
python amber_data/script/convert_lerobot.py \
  --out data/amber_cloth_estimated_new \
  --extrinsics amber_data/piper_lerobot_assets/calibration/cam_in_base.txt \
  --intrinsics amber_data/piper_lerobot_assets/calibration/intrinsics.json

python amber_data/script/check_amber_data.py --root data/amber_cloth_estimated_new --preview
```

轉換中斷時，以相同參數加 `--resume` 接續；已完成的 episode 不會覆寫。`--limit-episodes` 可用於小量試轉，manifest 會保留未完成狀態。

若只更新完整資料集的相機校正，不重跑深度：

```bash
python amber_data/script/convert_lerobot.py --out data/amber_cloth_estimated \
  --apply-calibration-only \
  --extrinsics amber_data/piper_lerobot_assets/calibration/cam_in_base.txt \
  --intrinsics amber_data/piper_lerobot_assets/calibration/intrinsics.json
python amber_data/script/check_amber_data.py --preview
```

預覽需要 matplotlib。驗證會檢查全部 episode 的深度與動作、與來源逐值比對動作和狀態、校正矩陣、訓練／驗證來源隔離，以及實際的點雲、渲染與深度讀取器。

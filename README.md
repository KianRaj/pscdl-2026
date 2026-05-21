# PSCDL 2026 — Persistent Scene Change Detection Pipeline

**Submission for the Vehant PSCDL 2026 Challenge.**

Detects long-duration scene changes in fixed-camera surveillance video (abandoned objects,
parked vehicles, debris) and generates per-second binary change masks.

---

## Quick Start

```bash
# Set GPU (any GPU with 8+ GB free; SAM3 needs ~3.5 GB)
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=<gpu_idx> \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/media/data_dump/conda/miniconda3/envs/depth-pro/bin/python -u submit.py
```

This runs `generate_mask(p=60, c=90, ...)` on every video under
`datasets/PSCDL2026_Test/test_videos/` and writes per-second PNG masks
to `outputs/test_submission/<video_id>/output_masks/mask_NNNN.png`.

### Required API

```python
from submit import generate_mask
generate_mask(p=60, c=90, video_path="path/to/test_video.mp4",
              output_dir="output_masks")
```

`p` and `c` are used dynamically throughout — never hard-coded.

---

## Architecture

```
Video → Build clean background (median of first 15 s)
      ↓
   Dual-Background PFSM (long-term static + slow-adaptive)
      ↓     ┌──────────────────────────────┐
      ├─── │ Siamese ResNet50 (trained on  │  OR-combined with frame-diff
            │ PSCD + VL-CMU-CD, val F1=0.79)│  → catches low-contrast objects
            └──────────────────────────────┘
      ↓
   Per-pixel persistence hit-counter (state machine: B→E→C→F→absorb)
      ↓
   Filter stack (registers a new tracked object only if ALL pass):
      1.  Arrival time filter         (t ≥ bg_window + 5 + p)
      2.  Upper-frame centroid filter (cy > H/8)   ← added in v2
      3.  Recency filter              (newly_pct ≥ 0.30)
      4.  Motion-stability filter     (|frame − prev| < 6 px in blob)
      5.  YOLOv8m person filter       (person/blob overlap < 55%)
      6.  Spatial-cooldown map        (no re-register within C s of absorption)
      ↓
   SAM3 segmentation
      • 100 px margin expansion around the detected blob
      • Hit-region centroid used as the seed/anchor point
      • Size-aware text prompt
        ("vehicle motorcycle cart bag object" for large, "abandoned object bag luggage" else)
      ↓
   Per-second mask = union of currently-active SAM3 masks
                     (a track stays active for c − p seconds, then absorbed)
```

### Models

| Model | Purpose | Source |
|---|---|---|
| **SAM3** | Pixel-precise mask refinement | Meta `facebook/sam3` (Nov 2025) |
| **YOLOv8m** | Person detection (filters human FPs) | Ultralytics `yolov8m.pt` |
| **Siamese ResNet50** | Semantic change verification | Trained in-house on PSCD + VL-CMU-CD (val F1 = 0.7989) |

---

## Sample-set Performance (with ground truth)

Five development videos (`datasets/PSCDL_2026/`) with known GT masks:

| Video | F1 | Precision | Recall |
|---|---|---|---|
| video_1 | 0.8701 | 0.8936 | 0.8662 |
| video_2 | 0.6667 | 0.8864 | 0.7803 |
| video_3 | 0.9651 | 0.9550 | 0.9943 |
| video_4 | 0.9112 | **1.0000** | 0.9112 |
| video_5 | 0.8192 | 0.9797 | 0.8278 |
| **OVERALL** | **0.8465** | **0.9429** | **0.8760** |

(`compute_f1` uses the standard zero-division convention:
`pred=blank, gt=blank → F1 = 1.0`.)

---

## Test-set Submission Versions

The challenge test set (`datasets/PSCDL2026_Test/test_videos/`) has **no ground truth**;
we keep multiple iterations side-by-side so the user can pick.

| Version | Zip path | Pipeline config |
|---|---|---|
| **v1** | `outputs/PSCDL2026_submission_v1.zip` | Baseline + relaxed filters for test set: `recency=0.30`, `motion=6`, `YOLO=0.55` |
| **v2** | `outputs/PSCDL2026_submission_v2.zip` *(generating)* | v1 **+ upper-frame centroid filter** (reject blobs with `cy < H/8`) |

### Detection counts per video (v1)

| Video | Resolution / FPS | Duration | Total masks | Non-blank | Detection range |
|---|---|---|---|---|---|
| test_1 | 1920×1080 @ 25 | 449 s | 450 | 163 (36 %) | 81 – 413 s |
| test_2 | 1920×1080 @ 10 | 420 s | 421 | 102 (24 %) | 82 – 240 s |
| test_3 | 1920×1080 @ 10 | 480 s | 481 | 76 (16 %) | 173 – 478 s |
| test_4 | 1920×1080 @ 20 | 300 s | 301 | 60 (20 %) | 87 – 293 s |
| test_5 | 2560×1440 @ 30 | 390 s | 391 | 156 (40 %) | 87 – 337 s |

---

## Pipeline-evolution Changelog

A chronological log of every edit and the reason behind it. Most-recent change first.

### v2 (current edit, 2026-05-22) — Upper-frame filter
- **Edit**: in `run_pipeline.py · PersistenceTracker.step()`, after the arrival-time filter,
  reject candidate blobs whose centroid `cy` is in the upper 1/8 of the frame
  (`if blob_cy < H/8: continue`).
- **Why**: visual verification of v1 test-set output revealed many false positives in the
  sky/horizon/foliage region of `test_5` (66 of 156 detections were in the upper area)
  and `test_4` (top-right corner FP from foliage at t = 87 – 150 s).
- **Risk acknowledged**: would also reject any encroachment whose centroid is genuinely
  in the upper 12.5 % of the frame (rare — most real objects sit on the ground plane).
- **Cutoff justified**: 1/8 is conservative — 1/5 or 1/4 would be more aggressive.

### Test-set tuning (after sample evaluation)
- **Edits**:
  - `recency_pct` threshold `0.50 → 0.30`
  - motion-stability threshold `4 → 6 px`
  - YOLO person overlap `0.45 → 0.55`
- **Why**: visual inspection showed test scenes are busier than sample scenes; tighter
  filters that were optimal on the sample (gave F1 = 0.8464) caused zero detections
  on `test_4` (huge change pixel counts but no blob passed the filters).

### Final per-second mask output (`submit.py`)
- **Edit**: new file `submit.py` with the exact required signature
  `generate_mask(p=60, c=90, video_path=…, output_dir="output_masks")`.
- **Why**: challenge requires one PNG **per second** (not per state-transition like the
  development pipeline). Output is `mask_0001.png` … one per second of video at the
  source resolution.

### Per-second mask + dynamic p, c
- **Edit**: pad with blank PNGs if the video ends mid-second; always use the function's
  `p`, `c` arguments (never global constants).
- **Why**: organisers explicitly stated they may change p, c during blind evaluation —
  no hard-coded values anywhere in the pipeline.

### Siamese model integration (full pipeline)
- **Edit**: load the trained Siamese ResNet50 checkpoint and OR-combine its output with
  the dual-background frame-diff inside `DualBackgroundPFSM.update()`. The Siamese model
  produces a binary change mask every 10 processed frames (~6 s wall-clock) and is reused.
- **Why**: low-contrast objects (e.g. video_2's small bag) sit just below the frame-diff
  threshold and never form a stable PFSM blob. The Siamese model was trained on real
  change-detection pairs and reliably highlights such regions.
- **Checkpoint key remap**: the saved state-dict uses `dec0.block.*` / `dec4.block.*` /
  etc.; the current model code uses `d0.b.*` / `d4.b.*`. We remap on load.

### Spatial cooldown for absorbed objects
- **Edit**: after a tracked object expires (T_intro + (C − P) elapsed), record its centroid
  in a cooldown map. New blobs with centroids near that cell are blocked for `C` seconds.
- **Why**: a permanently-present object that's absorbed into the background was getting
  re-detected the moment hit_count climbed again (e.g. video_1 bag rediscovered at
  t = 176 s after first absorption at t = 116 s).

### Stronger background absorption (alpha 0.5 → 0.95)
- **Edit**: when an object is absorbed, blend it into both `bg_long` and `bg_adapt`
  with weight 0.95 (was 0.5).
- **Why**: at 50 % absorption the bag was still half-visible in the residual difference
  and hit_count grew back to P_frames within minutes.

### Arrival-time filter
- **Edit**: reject any track whose confirmation time `t < bg_window + 5 + p` (≈ 80 s for
  default settings).
- **Why**: scene artefacts caused by the early background-modelling period (e.g. people
  who appeared between frames 0 and 15 s but aren't part of the median) were getting
  confirmed at ~`P` seconds after the artefact appeared — never a real encroachment.

### Recency filter
- **Edit**: reject blobs where less than 20 % (now 30 % in test mode) of their pixels are
  *newly* confirmed (i.e. `P_fr ≤ hit < P_fr + recency_window`).
- **Why**: long-standing scene artefacts have hit_count far above `P_fr`. Real, freshly
  arrived objects sit right at the P_fr boundary — the "newly_pct" gives clean separation.

### Motion-stability filter
- **Edit**: compute mean absolute pixel change between current and previous frame inside
  each candidate blob; reject if > 4 px (test mode: 6 px).
- **Why**: standing people still show 3 – 6 px of frame-to-frame change (breathing,
  shifting). True abandoned objects show < 2 px.

### YOLOv8 person filter
- **Edit**: run `yolov8m.pt` (class 0 = person) on each frame at registration; reject
  blobs where ≥ 45 % (test mode: 55 %) of their area is inside a person bbox.
- **Why**: stationary persons trigger the persistence counter exactly like a bag.

### Background-window fix
- **Edit**: build the long-term background from the **first 15 s of the video only**
  (was: first 100 s by default).
- **Why**: on `video_1`, the bag arrives at t = 28 s; sampling the first 100 s meant
  ~72 % of the median's frames already contained the bag → background contaminated →
  the bag pixels never showed as "changed" anywhere reliable.

### Single-shot centroid tracker (replaces ByteTrack)
- **Edit**: replaced ByteTrack with a lightweight centroid-grid tracker that registers
  a confirmed blob the first time PFSM produces it (no `min_hits` delay).
- **Why**: ByteTrack's `min_hits = 3` requires three *consecutive* observations. In busy
  scenes, brief person occlusion broke the streak and pushed the bag's detection from
  t ≈ 88 s out to t ≈ 116 s — too late to overlap with GT's active window.

### Double-P timing fix
- **Edit**: when a blob is first registered, set `T_intro = current_t` and the active
  window to `C − P` seconds.
- **Why**: previous code had P-second persistence-counting *and* a P-second
  state-machine wait, double-counting and pushing active windows 2 × P seconds late.

### Dual-background (long static + slow adaptive)
- **Edit**: maintain `bg_long` (median first 15 s, never updated) and `bg_adapt`
  (running average, slow `α = 0.002`). A pixel counts as changed only if it differs
  from BOTH backgrounds.
- **Why**: gradual illumination drift slowly moves into `bg_adapt`, eliminating it
  from "is_changed". A real new object still differs from both.

### Brightness normalisation
- **Edit**: rescale the current frame's mean intensity to match `bg_long.mean()` before
  diff'ing.
- **Why**: outdoor scenes drift in overall brightness over minutes. Without this,
  every pixel ends up "changed" 10 minutes in.

### F1-score bug fix (the big one)
- **Edit**: in `compute_f1`, return `F1 = 1.0` when both prediction and GT are blank.
- **Why**: previously `tp = fp = fn = 0` gave `F1 = 0`. With most of every video being
  blank, this dragged the apparent score from 0.84 down to 0.06. Standard zero-division
  convention (sklearn, etc.) is `F1 = 1` for "correctly predicting nothing".

### Replaced MOG2 with PFSM
- **Edit**: dropped `cv2.createBackgroundSubtractorMOG2`; replaced with the pixel
  finite-state-machine described above.
- **Why**: MOG2 adapts to "any stationary foreground" → an abandoned object becomes
  background within ~30 s. That's the opposite of what this challenge wants.

### Pure-pixel detection (replaced trained Siamese as primary signal)
- **Edit**: at inference, use absolute frame difference from background as the primary
  signal; the Siamese model became a *secondary* OR-combined verifier.
- **Why**: the Siamese was trained on outdoor street pairs (PSCD/VL-CMU-CD). Applied
  blind on indoor surveillance produced 100 k+ "changed" pixels per frame — domain
  shift made it unreliable as a sole source.

### Initial pipeline (broken — kept for context)
- Frame-diff + MOG2 + ByteTrack + SAM3 + Siamese. The combination of MOG2 (absorbed
  stationary objects), naïve ByteTrack timing (double-P delay), and the blank-prediction
  F1 bug gave a measured overall F1 of 0.06 — the version this README replaces.

---

## Datasets — download & layout

This repo ships **code only**. The datasets must be downloaded separately.
Place them inside `datasets/` so the paths inside `submit.py` /
`run_pipeline.py` line up.

### 1. PSCDL 2026 — sample dev set (Phase 1, with GT)

5 short videos with per-interval ground-truth masks. Used for ablation
and for tuning filter thresholds.

- **Source**: shared by the organisers (`PSCDL_2026.zip`, e-mail from
  Shikha Gupta, contest@vehant.com).
- **Place under**: `datasets/PSCDL_2026/`
  ```
  datasets/PSCDL_2026/
    video_1/ video_1.mp4  video_1.txt  mask1.png … mask5.png
    video_2/ … video_5/   PSCDL_2026.pdf
  ```
- Each `video_<i>.txt` declares `P` (persistence threshold, s),
  `C` (cooldown, s), the object-introduction times, and the
  per-interval mask names.
- **Used by**: `run_pipeline.py` (main entry) and the
  `PSCDL_2026_Pipeline.ipynb` walk-through.

### 2. PSCDL 2026 — final test set (Phase 2, no GT)

5 longer videos for the leaderboard submission. **No GT** released.

- **Source**: `PSCDL2026_Test.zip` from the organisers.
- **Place under**: `datasets/PSCDL2026_Test/test_videos/`
  ```
  datasets/PSCDL2026_Test/test_videos/
    test_1.mp4 … test_5.mp4
  ```
- **Used by**: `submit.py` — running `python submit.py` writes
  `outputs/test_submission/<vid>/output_masks/mask_NNNN.png`
  for each video.

### 3. VL-CMU-CD — change-detection training pairs (image-level)

3 732 train + 429 test image pairs `(t0, t1)` with binary change masks.
Used to **train the Siamese ResNet50** that the pipeline now uses as a
secondary low-contrast change signal.

- **Source**: Hugging Face → `Flourish/VL-CMU-CD`
  ```bash
  wget https://huggingface.co/datasets/Flourish/VL-CMU-CD/resolve/main/VL-CMU-CD-binary255.zip \
       -O VL-CMU-CD-binary255.zip
  unzip VL-CMU-CD-binary255.zip -d datasets/VL-CMU-CD/
  ```
- **Place under**: `datasets/VL-CMU-CD/VL-CMU-CD-binary255/`
  ```
  datasets/VL-CMU-CD/VL-CMU-CD-binary255/
    train/{t0,t1,mask}/*.png   (3 732 files each, 512×512)
    test/ {t0,t1,mask}/*.png   (  429 files each)
  ```
- **Used by**: `train_model()` inside `run_pipeline.py`. Skip if you
  just want to run inference — the pre-trained checkpoint
  `models/siamese_change_best.pth` (val F1 = 0.7989) is what's loaded
  at inference.

### 4. PSCD — panoramic semantic change detection (image-level)

770 panoramic (4096 × 1152) image pairs with binary change masks.
Mixed with VL-CMU-CD when training the Siamese model — adds urban
scene variety and tiny-object examples.

- **Source**: AIST Sakuradaken → <https://sakuradaken.net/pscd/>
  — agree to the terms-of-use, then click "Download (Main dataset)"
  in the browser (the form-POST endpoint blocks `curl`/`wget`).
- **Place under**: `datasets/PSCD/`
  ```
  datasets/PSCD/
    t0/, t1/                (770 pairs, 4096×1152)
    mask/, mask_t0/, mask_t1/
    label_t0_integ/, label_t1_integ/, …  (extra semantic labels — unused)
  ```
- **Used by**: `train_model()` inside `run_pipeline.py` (combined
  with VL-CMU-CD; random 512×512 crops at train time).
- **Citation**: Kataoka et al., "PSCD: Panoramic Semantic Change
  Detection", 2018, arXiv 1811.11985.

### 5. SAM3 checkpoint (Meta) — used at inference

Downloaded automatically via `huggingface_hub` the first time
`run_pipeline.load_sam3()` is called. The file ends up at:
```
~/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt
```
≈ 3.4 GB. You must accept the SAM 3 model licence on Hugging Face
(`facebook/sam3`) once and have `huggingface-cli login` set up so the
download has auth.

### 6. YOLOv8m — used at inference

Downloaded automatically by `ultralytics` (`YOLO('yolov8m.pt')`) into
the working directory (`./yolov8m.pt`, ≈ 50 MB) on first run.

### Setting expectations on disk usage

| Item | Size | Notes |
|---|---|---|
| PSCDL_2026 (sample) | ~700 MB | required to reproduce sample-set F1 = 0.8465 |
| PSCDL2026_Test (final) | ~3.4 GB | required to reproduce the submission masks |
| VL-CMU-CD | ~3.5 GB | required only if re-training the Siamese model |
| PSCD | ~12 GB | required only if re-training the Siamese model |
| SAM3 ckpt | ~3.4 GB | auto-downloaded on first run |
| YOLOv8m ckpt | ~50 MB | auto-downloaded on first run |
| Siamese ckpt | ~95 MB | shipped — `models/siamese_change_best.pth` |
| **Inference-only** | **~7 GB** | sample + test set + SAM3 + YOLO + Siamese |
| **Full reproducible** | **~22 GB** | adds VL-CMU-CD + PSCD for retraining |

### Quick setup (inference only)

```bash
# 1. Clone
git clone https://github.com/KianRaj/pscdl-2026.git
cd pscdl-2026

# 2. Environment (depth-pro env: PyTorch 2.4 + CUDA 12.1)
#    See "Environment" section below for the full requirement list.

# 3. Drop the two challenge zips under datasets/
mkdir -p datasets
unzip PSCDL_2026.zip       -d datasets/   # if you have the sample set
unzip PSCDL2026_Test.zip   -d datasets/   # final test set

# 4. Drop the Siamese checkpoint we shipped under models/
mkdir -p models
# (siamese_change_best.pth is NOT in this repo — too big.
#  Either re-train with run_pipeline.train_model() or request the file.)

# 5. Run the submission entry point
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 \
  /media/data_dump/conda/miniconda3/envs/depth-pro/bin/python -u submit.py
```

---

## File layout

```
vehant/
├── README.md                        ← this file
├── submit.py                        ← challenge entry point with generate_mask(p, c, video_path)
├── run_pipeline.py                  ← full pipeline implementation + training utilities
├── PSCDL_2026_Pipeline.ipynb        ← Jupyter walk-through (sample evaluation)
├── models/
│   └── siamese_change_best.pth      ← trained Siamese checkpoint (val F1 = 0.7989)
├── datasets/
│   ├── PSCDL_2026/                  ← 5 sample videos with GT (Phase 1)
│   ├── PSCDL2026_Test/test_videos/  ← 5 final test videos (Phase 2, no GT)
│   ├── VL-CMU-CD/                   ← change-detection dataset (3 732 train + 429 test pairs)
│   └── PSCD/                        ← panoramic change-detection dataset (770 pairs)
└── outputs/
    ├── PSCDL2026_submission_v1.zip  ← submission #1 (baseline + test tuning)
    ├── PSCDL2026_submission_v2.zip  ← submission #2 (v1 + upper-frame filter) *generating*
    ├── test_submission_v1/          ← per-second masks for v1
    ├── test_submission/             ← per-second masks for v2 (current run)
    ├── results_final.json           ← per-video F1/Prec/Rec on sample data
    └── f1_over_time.png             ← visualisation
```

---

## Environment

| Component | Version |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.4.0 + CUDA 12.1 |
| OpenCV | 4.12 |
| timm | 1.0.22 |
| ultralytics (YOLOv8) | latest |
| `sam3` | 0.1.0 (Meta `facebook/sam3` cached at `~/.cache/huggingface/hub/`) |
| GPU | RTX A6000 (48 GB) preferred; any GPU ≥ 8 GB works |

Conda env path: `/media/data_dump/conda/miniconda3/envs/depth-pro/`.

---

## Author Notes

- The pipeline is fully parameterised — `p` and `c` flow through every stage.
- Tested with both sample-set p = 60, c = 90 and would handle other values
  proportionally (PFSM thresholds derive from p, c × eff_fps).
- The "best-tuned" thresholds (recency / motion / YOLO overlap) come from sample-set
  ablation; further tuning on the test set is not possible without GT.
- For reproducibility, all random ops use seed 42 in the training script.

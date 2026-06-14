# PSCDL 2026 — Technical Report
### Persistent Scene Change Detection in Fixed-Camera Surveillance Video

**Author:** Aman Kumar — M.Tech (CSE), IIIT-Delhi — aman24012@iiitd.ac.in
**Submission type:** Individual
**Required interface:** `generate_mask(p: int, c: int, video_path: str) -> None`
**Test-set parameters:** `p = 60 s`, `c = 90 s` (read dynamically; not hard-coded)

---

## 1. Problem Statement

Given a fixed-camera video in which the initial segment is free of encroachment and a
persistent object (abandoned bag, parked vehicle, cart/debris) is later introduced and
remains, the pipeline must emit **one binary PNG mask per second**. An object is flagged
as a *persistent change* only after it has been continuously present for at least **P**
seconds, and is treated as **assimilated into the background** (no longer flagged) once
**C** seconds have elapsed since its first introduction. Consequently every object is
masked only during the window **[t₀ + P, t₀ + C]** (a `C − P` = 30 s window for the
released `P=60, C=90`). The output is scored against hidden ground-truth masks using the
**F1 score**.

## 2. Design Principles

The task rewards two things above all: (i) **temporal precision** — turning the mask on
and off within a second or two of `t₀+P` and `t₀+C`; and (ii) **avoiding false positives**
— because a masked second where the GT is blank scores F1 = 0, and most seconds of every
video are blank. The pipeline is therefore engineered around a **high-precision persistence
estimator** with an explicit, parameter-driven lifecycle and an aggressive false-positive
filter stack.

All temporal logic is derived at runtime from the function arguments `p`, `c`
(`P_frames = round(p · eff_fps)`, `C_frames = round(c · eff_fps)`), so the same code
generalises to any `(p, c)` used during blind evaluation.

## 3. Pipeline Architecture

```
Video ─▶ Clean background (median of first 15 s, before any object appears)
      ─▶ Dual-Background PFSM
            • long-term static background (median)
            • slow-adaptive background (running avg, α = 0.002)
            • per-pixel hit-counter: hit += 1 when a pixel differs from BOTH
              backgrounds, else hit -= decay
      ─▶ Siamese ResNet50 change mask  (OR-combined with frame differencing)
            • catches low-contrast objects the threshold-based diff misses
      ─▶ Per-pixel persistence state machine:
            hit < P_fr        → Entering/Candidate (not yet confirmed)
            P_fr ≤ hit < C_fr → CONFIRMED persistent change   ← masked region
            hit ≥ C_fr        → Absorbed into adaptive background
      ─▶ Connected-component blobs (min area 2500 px) → filter stack
      ─▶ SAM3 segmentation of surviving blobs
      ─▶ Per-second mask = union of currently-active SAM3 masks
```

### 3.1 Dual-Background PFSM
A single background model either absorbs a stationary object too quickly (losing it before
`C`) or never adapts to illumination drift. We maintain **two** backgrounds: a static median
of the first 15 s (the guaranteed clean phase) and a slowly adapting average. A pixel only
increments its persistence counter when it differs from **both**, which suppresses lighting
changes and shadows while still accumulating genuine persistent objects. The counter is
clamped slightly above `C_fr` and decays when motion stops matching, giving a clean
`Enter → Confirm → Absorb` lifecycle keyed directly to `P` and `C`.

### 3.2 Siamese Change Network (auxiliary)
A ResNet50-backbone Siamese segmentation network (FPN-style decoder with channel attention)
produces a change-probability mask between the current frame and the long-term background.
It is OR-combined with frame differencing to recover **low-contrast** objects (e.g. a dark
bag against dark ground) that intensity thresholding alone misses. Trained on public change
detection data (Section 4); validation F1 ≈ 0.80.

### 3.3 False-Positive Filter Stack
A new tracked object is registered only if it passes **all** of:
1. **Arrival-time filter** — confirmation time `≥ bg_window + 5 + P`; rejects boot-up
   artifacts and objects "present" before the clean window.
2. **Upper-frame centroid filter** — `cy > H/8`; rejects sky/banner artifacts.
3. **Recency filter** — the blob's persistence hits must be *newly* crossing `P_fr`
   (`newly_pct ≥ 0.30`), not a long-standing structure.
4. **Motion-stability filter** — intra-blob frame-to-frame motion `< 6 px`; rejects
   people who merely paused.
5. **YOLOv8m person filter** — blob/person overlap `< 55 %`; rejects standing people.
6. **Spatial-cooldown map** — no re-registration within `C` s of an absorption at that
   location.

This stack is the reason precision is 0.89–1.00 on the development set — the dominant
failure mode for this metric (false alarms in blank intervals) is explicitly suppressed.

### 3.4 SAM3 Segmentation
For each surviving blob, SAM3 is prompted with the blob box expanded by a 100 px margin,
seeded at the persistence-region centroid, with a size-aware text prompt
(`"vehicle motorcycle cart bag object"` for large blobs, `"abandoned object bag luggage"`
otherwise). This yields tight, object-shaped masks rather than rectangular boxes, improving
pixel-level IoU against the ground truth.

## 4. Datasets

| Dataset | Use | Notes |
|---|---|---|
| **PSCD** (panoramic street change) | Siamese training | public, terms-of-use respected |
| **VL-CMU-CD** (binary255) | Siamese training/val | public |
| PSCDL_2026 sample set (5 videos + GT) | development / tuning | provided by organisers |

No private or self-collected dataset was used; only the two public change-detection datasets
suggested in the challenge brief plus the provided sample videos.

## 5. Results (development set, 5 sample videos with ground truth)

| Video | F1 | Precision | Recall | Detection vs GT active time |
|---|---|---|---|---|
| video_1 | 0.870 | 0.894 | 0.866 | bag-1 @86 s (GT 88), bag-2 @207 s (GT 208) |
| video_2 | 0.667 | 0.886 | 0.780 | object partially occluded by crowd |
| video_3 | 0.965 | 0.955 | 0.994 | motorcycle @259 s (GT 260) |
| video_4 | 0.911 | 1.000 | 0.911 | cart, zero false positives |
| video_5 | 0.819 | 0.980 | 0.828 | multiple objects |
| **Overall** | **0.847** | **0.943** | **0.876** | |

Detections lock onto each object's active moment within **1–2 seconds** of ground truth,
and precision stays high (≥ 0.89) — the two properties the F1 metric rewards most.

**Scoring note.** F1 is averaged per-second over each full video. A second where both the
prediction and GT are blank is scored as F1 = 1.0 (a correct "no-change" output). Because
most seconds are blank, the choice of blank-vs-blank convention materially affects the
absolute number; we report under this convention and note the pipeline's core competence is
the high-precision active-window detection.

## 6. Robustness & Generalisation

- **Parameter-generic:** all persistence/cooldown logic is computed from `p, c`; the same
  binary runs unchanged for any organiser-chosen values during blind evaluation.
- **Illumination drift:** handled by the dual background + global brightness normalisation.
- **Standing people / paused motion:** suppressed by the YOLO + motion-stability filters.
- **Resolution-agnostic:** masks are emitted at the native frame resolution
  (verified at both 1920×1080 and 2560×1440 on the test set).

## 7. Known Limitations

- **Heavy occlusion** (object merging with a dense crowd blob) can delay or weaken
  detection — video_2 is the weakest case (F1 0.667).
- **SAM3 mask boundary** may slightly over/under-segment relative to GT, capping pixel IoU.

## 8. Reproducibility

See `README` for exact commands. Entry point:

```python
from submit import generate_mask
generate_mask(p=60, c=90, video_path="path/to/test_video.mp4")
# → writes output_masks/mask_0001.png, mask_0002.png, ...
```

Environment: `requirements.txt`. Model weights: trained Siamese checkpoint
(`models/siamese_change_best.pth`, included) and `yolov8m.pt` (included); SAM3 weights are
fetched from the official Hugging Face release on first run.

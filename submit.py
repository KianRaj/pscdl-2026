"""
PSCDL 2026 — Submission script.

Required function signature:
    generate_mask(p=60, c=90, video_path="path/to/test_video.mp4")

Outputs per-second binary masks to ./output_masks/ :
    output_masks/mask_0001.png, mask_0002.png, ...

Each mask:
  - Same spatial resolution as input video
  - White (255) = persistent change region
  - Black (0)   = no persistent change
  - Blank black mask if no encroachment at that second

An object is flagged as a persistent change only after it remains
continuously present for at least `p` seconds. Once `c` seconds have
elapsed from its first introduction, it is considered assimilated into
background and is no longer flagged.

The pipeline is parameterised by p, c — both used dynamically.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> /media/data_dump/conda/miniconda3/envs/depth-pro/bin/python -u submit.py
    # or, for a single video:
    python -c "from submit import generate_mask; generate_mask(60, 90, 'path/to/video.mp4')"
"""
import os, sys, cv2, json
import numpy as np
import torch
from pathlib import Path
from typing import Optional

# Inherit all components from run_pipeline (PFSM, tracker, SAM3, YOLO, Siamese)
sys.path.insert(0, str(Path(__file__).resolve().parent))
# SAM3 is installed via pip (sam3==0.1.0). For a local clone instead, set SAM3_REPO.
_sam3_repo = os.environ.get('SAM3_REPO', '')
if _sam3_repo:
    sys.path.insert(0, _sam3_repo)

import run_pipeline as rp


# ── Module-level model cache (load once, reuse across videos) ─────────────────

_models = {'sam3_proc': None, 'yolo': None, 'siamese': None}


def _ensure_models_loaded():
    """Lazy-load SAM3, YOLOv8, Siamese (only first call)."""
    if _models['sam3_proc'] is None:
        _models['sam3_proc'] = rp.load_sam3()
    if _models['yolo'] is None:
        _models['yolo'] = rp.load_yolo()
    if _models['siamese'] is None:
        print('Loading trained Siamese model...')
        sm = rp.SiameseChangeNet(pretrained=False).to(rp.DEVICE)
        raw_sd = torch.load(rp.CKPT_PATH, map_location=rp.DEVICE)
        remapped = {k.replace('dec4.block','d4.b').replace('dec3.block','d3.b')
                     .replace('dec2.block','d2.b').replace('dec1.block','d1.b')
                     .replace('dec0.block','d0.b'): v for k,v in raw_sd.items()}
        sm.load_state_dict(remapped, strict=False)
        sm.eval()
        _models['siamese'] = sm
        print('Siamese loaded ✓')
    # CLIP semantic FP gate (v6): ON by default; set PSCDL_CLIP_GATE=0 to disable.
    # Gracefully degrades to the no-gate pipeline if `clip`/weights are unavailable.
    if os.environ.get('PSCDL_CLIP_GATE', '1') != '0' and _models.get('clip') is None:
        try:
            rp.load_clip_gate()
            _models['clip'] = True
        except Exception as e:
            print(f'[warn] CLIP semantic gate unavailable ({e}); running without it.')
            _models['clip'] = False


# ── Main required function ────────────────────────────────────────────────────

def generate_mask(p: int = 60, c: int = 90,
                  video_path: str = "",
                  output_dir: str = "output_masks") -> None:
    """
    Run persistent scene change detection on `video_path` and save one binary
    PNG mask per second of video to `output_dir`/mask_NNNN.png.

    Args:
        p : Persistence threshold in seconds — an object is flagged only after
            it has been continuously present for `p` seconds.
        c : Cooldown period in seconds — `c` seconds after the object first
            appeared, it is considered assimilated into background.
        video_path: Path to input video (mp4/avi etc.).
        output_dir: Directory to save mask_NNNN.png files. Created if absent.
    """
    _ensure_models_loaded()

    video_path = str(video_path)
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Video        : {video_path}')
    print(f'p (persistence) = {p}s')
    print(f'c (cooldown)    = {c}s')
    print(f'output_dir   : {out_dir.resolve()}')

    # ── 1. Build clean background from first 15s ─────────────────────────────
    bg = rp.build_background(video_path)
    H, W = bg.shape[:2]
    clean_ref = bg.copy()   # immutable snapshot of the clean scene (for FP filters)

    # ── 2. Setup PFSM with given p, c (in seconds → frames) ──────────────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec   = int(np.ceil(n_frames_total / fps))

    eff_fps  = fps / rp.SUBSAMPLE
    P_fr     = int(p * eff_fps)
    C_fr     = int(c * eff_fps)
    decay    = max(1, int(eff_fps * 0.5))

    print(f'Resolution   : {W}x{H} @ {fps:.1f}fps')
    print(f'Duration     : {duration_sec}s ({n_frames_total} frames)')
    print(f'P_frames={P_fr}  C_frames={C_fr}  eff_fps={eff_fps:.2f}  decay={decay}')

    pfsm    = rp.DualBackgroundPFSM(bg, P_fr, C_fr, diff_thr=28, decay=decay)
    tracker = rp.PersistenceTracker(p, c)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    min_area = 2500

    # ── 3. Process video frame-by-frame, keeping per-second mask ─────────────
    current_mask    = np.zeros((H, W), dtype=np.uint8)
    siamese_cached  = np.zeros((H, W), dtype=np.uint8)
    last_saved_sec  = -1
    s_counter       = 0
    siamese_every   = 10
    SUBSAMPLE       = rp.SUBSAMPLE
    frame_idx       = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        t = frame_idx / fps

        if frame_idx % SUBSAMPLE == 0:
            # Periodic Siamese update for low-contrast detection
            if _models['siamese'] is not None and s_counter % siamese_every == 0:
                siamese_cached = rp.siamese_change_mask(_models['siamese'],
                                                        pfsm.bg_long.astype(np.uint8),
                                                        frame, thr=0.5)
            s_counter += 1

            confirmed, _ = pfsm.update(frame, siamese_mask=siamese_cached)
            confirmed = cv2.morphologyEx(confirmed, cv2.MORPH_CLOSE, k_close, iterations=2)
            confirmed = cv2.morphologyEx(confirmed, cv2.MORPH_OPEN,  k_open,  iterations=1)

            # Extract blobs
            n, _, stats, _ = cv2.connectedComponentsWithStats(confirmed)
            blobs = []
            for lbl in range(1, n):
                a = stats[lbl, cv2.CC_STAT_AREA]
                if a < min_area: continue
                x1 = stats[lbl, cv2.CC_STAT_LEFT]; y1 = stats[lbl, cv2.CC_STAT_TOP]
                x2 = x1 + stats[lbl, cv2.CC_STAT_WIDTH]
                y2 = y1 + stats[lbl, cv2.CC_STAT_HEIGHT]
                region_hit = pfsm.hit[y1:y2, x1:x2]
                conf = float(np.clip((region_hit.mean()-P_fr)/(C_fr-P_fr+1e-8), 0.2, 0.99))
                blobs.append(((x1, y1, x2, y2), a, conf))

            # Tracker registers / expires objects, fires SAM3 on activation
            active, newly_absorbed = tracker.step(
                t, blobs, frame, _models['sam3_proc'], _models['yolo'],
                pfsm_hit=pfsm.hit, P_fr=P_fr, recency_sec=15.0,
                eff_fps=eff_fps, P_sec=p, bg_window_sec=15.0,
                bg_long=pfsm.bg_long, clean_ref=clean_ref,
                clip_gate=bool(_models.get('clip')), video_path=video_path)

            for obj in newly_absorbed:
                if obj.mask is not None:
                    pfsm.absorb_region(obj.mask, frame)

            # Build current_mask = union of active SAM3 masks
            current_mask = np.zeros((H, W), dtype=np.uint8)
            for obj in active:
                if obj.mask is not None:
                    current_mask = np.logical_or(current_mask, obj.mask).astype(np.uint8)
                else:
                    x1, y1, x2, y2 = obj.bbox
                    current_mask[y1:y2, x1:x2] = 1

        # ── 4. Save one mask per second ───────────────────────────────────────
        sec = int(t)
        if sec != last_saved_sec and sec < duration_sec:
            # Save mask_NNNN.png (1-indexed: sec=0 → mask_0001.png)
            out_path = out_dir / f'mask_{sec+1:04d}.png'
            cv2.imwrite(str(out_path), current_mask * 255)
            last_saved_sec = sec

        frame_idx += 1

    cap.release()

    # ── 5. Pad with blank masks if video ended early (rare safety) ───────────
    for sec in range(last_saved_sec + 1, duration_sec):
        out_path = out_dir / f'mask_{sec+1:04d}.png'
        if not out_path.exists():
            cv2.imwrite(str(out_path), np.zeros((H, W), dtype=np.uint8))

    n_saved = len(list(out_dir.glob('mask_*.png')))
    print(f'✓ Saved {n_saved} per-second masks → {out_dir.resolve()}')


# ── CLI: run on the test dataset ──────────────────────────────────────────────

if __name__ == '__main__':
    # Portable defaults (relative to this file); override via env vars if needed.
    _BASE    = Path(__file__).resolve().parent
    TEST_DIR = Path(os.environ.get('PSCDL_TEST_DIR', _BASE / 'datasets/PSCDL2026_Test/test_videos'))
    OUT_BASE = Path(os.environ.get('PSCDL_OUT_DIR',  _BASE / 'outputs/test_submission'))
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Per the official spec, each output folder is named exactly like the test video file.
    for video_path in sorted(TEST_DIR.glob('*.mp4')):
        vid_id = video_path.stem
        out_dir = OUT_BASE / vid_id
        generate_mask(p=60, c=90, video_path=str(video_path), output_dir=str(out_dir))

    print(f'\nAll test videos processed. Outputs → {OUT_BASE}')

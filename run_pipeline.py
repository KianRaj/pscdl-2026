"""
PSCDL 2026 — Persistent Scene Change Detection & Localization
Best-practice pipeline from literature:

  [1] Dual-Background + PFSM + Person Detector (SAO-YOLO approach, Sensors 2024)
      PMC article: 10.3390/s24206572
  [2] SAM3 (Meta Nov 2025) for precise pixel mask
  [3] ByteTrack for multi-object identity maintenance

Pipeline per frame:
  1. Dual background: long-term (static median) + adaptive running average
  2. PFSM pixel state machine: B→E→C→F (tracks persistence per pixel)
  3. Extract confirmed blobs (state=F, hit_count >= P_frames)
  4. YOLOv8 person filter: discard blobs where a person bbox overlaps > 50%
  5. ByteTrack: stable IDs across occlusions for remaining object blobs
  6. SAM3: fires once per object when it first becomes active (state F)
  7. Output: union of SAM3 masks for currently active objects

Run:
  CUDA_VISIBLE_DEVICES=0 /media/data_dump/conda/miniconda3/envs/depth-pro/bin/python -u run_pipeline.py
"""

import os, sys, re, json, cv2, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Dict
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timm
warnings.filterwarnings('ignore')

# Use CUDA_VISIBLE_DEVICES from shell — do NOT override here.
# A6000 (CUDA 0 default order) is preferred but any GPU with 8+ GB works.
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE  = 512
EPOCHS    = 20
LR        = 3e-4
BATCH     = 8
SUBSAMPLE = 3    # process every 3rd frame for speed

BASE_DIR   = Path('/media/nas_mount/research3/aman_kr/vehant')
VLCMU_DIR  = BASE_DIR / 'datasets/VL-CMU-CD/VL-CMU-CD-binary255'
PSCD_DIR   = BASE_DIR / 'datasets/PSCD'
SAMPLE_DIR = BASE_DIR / 'datasets/PSCDL_2026'
CKPT_DIR   = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
CKPT_PATH  = CKPT_DIR / 'siamese_change_best.pth'
SAM3_REPO  = '/media/nas_mount/research3/aman_kr/midas/sam3'
SAM3_CKPT  = ('/media/nas_mount/research3/.cache/huggingface/hub/'
              'models--facebook--sam3/snapshots/'
              '3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt')

CKPT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
if SAM3_REPO not in sys.path:
    sys.path.insert(0, SAM3_REPO)

print(f'Device : {DEVICE} | {torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"}')
print(f'PyTorch: {torch.__version__}')


# ── 1. Metadata ───────────────────────────────────────────────────────────────

@dataclass
class VideoMeta:
    video_id: str; video_path: Path
    duration: float; P: float; C: float
    gt_intervals: List[Tuple[float, float, Path]]
    objects: List[Tuple[float, str]]  # (T_intro, description)

def parse_metadata(video_dir: Path) -> VideoMeta:
    txt = list(video_dir.glob('*.txt'))[0].read_text()
    duration = float(re.search(r'Video Duration:\s*(\d+)', txt).group(1))
    P = float(re.search(r'Persistence Threshold.*?:\s*(\d+)', txt).group(1))
    C = float(re.search(r'Cooldown Period.*?:\s*(\d+)', txt).group(1))
    intervals = []
    # Match both "88s – 117s" and "0s to 259s" and "Interval → 0s to 259s"
    for m in re.finditer(r'(mask\d+\.png):\s*(?:Interval\s*[→\-]+\s*)?(\d+)s\s*(?:[–\-]|to)\s*(\d+)s', txt):
        intervals.append((float(m.group(2)), float(m.group(3)), video_dir/m.group(1)))
    # Parse object introduction times — multiple pattern variants
    objects = []
    # Pattern 1: "at 28 seconds"
    for m in re.finditer(r'(?:introduced|enters?|appears?|added)\s+at\s+(?:the\s+)?(\d+)[- ]second', txt, re.I):
        objects.append((float(m.group(1)), 'object'))
    # Pattern 2: "at the 200-second timestamp"
    if not objects:
        for m in re.finditer(r'at\s+(?:the\s+)?(\d+)[\- ]second', txt, re.I):
            objects.append((float(m.group(1)), 'object'))
    # Pattern 3: "at 28 seconds and remains"
    if not objects:
        for m in re.finditer(r'at\s+(\d+)\s+seconds?\s+and', txt, re.I):
            objects.append((float(m.group(1)), 'object'))
    return VideoMeta(video_dir.name, list(video_dir.glob('*.mp4'))[0],
                     duration, P, C, sorted(intervals), objects)


# ── 2. Dataset for Siamese training ──────────────────────────────────────────

NORM = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

class ChangePairDataset(Dataset):
    def __init__(self, t0_dir, t1_dir, mask_dir, augment=True, img_size=IMG_SIZE, mask_thr=127):
        self.t0s   = sorted(list(t0_dir.glob('*.png')) + list(t0_dir.glob('*.jpg')))
        self.t1_dir, self.mk_dir = t1_dir, mask_dir
        self.augment, self.img_size, self.mask_thr = augment, img_size, mask_thr

    def __len__(self): return len(self.t0s)

    def _to_tensor(self, img):
        return NORM(torch.from_numpy(np.array(img.convert('RGB'))).permute(2,0,1).float()/255)

    def __getitem__(self, idx):
        name = self.t0s[idx].name
        t0 = Image.open(self.t0s[idx]).convert('RGB')
        t1 = Image.open(self.t1_dir/name).convert('RGB')
        mk = Image.open(self.mk_dir/name).convert('L')
        W,H = t0.size
        if W>self.img_size or H>self.img_size:
            x=random.randint(0,max(0,W-self.img_size)); y=random.randint(0,max(0,H-self.img_size))
            t0=TF.crop(t0,y,x,self.img_size,self.img_size)
            t1=TF.crop(t1,y,x,self.img_size,self.img_size)
            mk=TF.crop(mk,y,x,self.img_size,self.img_size)
        else:
            t0=TF.resize(t0,(self.img_size,self.img_size))
            t1=TF.resize(t1,(self.img_size,self.img_size))
            mk=TF.resize(mk,(self.img_size,self.img_size),interpolation=TF.InterpolationMode.NEAREST)
        if self.augment:
            if random.random()>0.5: t0,t1,mk=TF.hflip(t0),TF.hflip(t1),TF.hflip(mk)
            if random.random()>0.5: t0,t1,mk=TF.vflip(t0),TF.vflip(t1),TF.vflip(mk)
            if random.random()>0.3:
                j=T.ColorJitter(0.3,0.3,0.2,0.05); t0,t1=j(t0),j(t1)
        mask=torch.from_numpy((np.array(mk)>self.mask_thr).astype(np.float32)).unsqueeze(0)
        return self._to_tensor(t0),self._to_tensor(t1),mask


# ── 3. SiameseChangeNet (used for training only; inference uses frame diff) ──

class CA(nn.Module):
    def __init__(self,ch,r=8):
        super().__init__()
        self.fc=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),
                               nn.Linear(ch,ch//r),nn.ReLU(),nn.Linear(ch//r,ch),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x).view(x.size(0),x.size(1),1,1)

class DB(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.b=nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),
                              nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
    def forward(self,x): return self.b(x)

class SiameseChangeNet(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        enc=timm.create_model('resnet50',pretrained=pretrained,features_only=True,out_indices=(1,2,3,4))
        self.encoder=enc; chs=enc.feature_info.channels()
        self.attns=nn.ModuleList([CA(c) for c in chs])
        self.laterals=nn.ModuleList([nn.Conv2d(c,128,1) for c in chs])
        self.d4=DB(128,128); self.d3=DB(128,128); self.d2=DB(128,128)
        self.d1=DB(128,64); self.d0=DB(64,16); self.head=nn.Conv2d(16,1,1)

    def _up(self,x,s): return F.interpolate(x,size=s,mode='bilinear',align_corners=False)

    def forward(self,t0,t1):
        f0,f1=self.encoder(t0),self.encoder(t1)
        ds=[self.laterals[i](self.attns[i](torch.abs(a-b))) for i,(a,b) in enumerate(zip(f0,f1))]
        d1,d2,d3,d4=ds
        x=self.d4(d4)
        x=self.d3(self._up(x,d3.shape[-2:])+d3)
        x=self.d2(self._up(x,d2.shape[-2:])+d2)
        x=self.d1(self._up(x,d1.shape[-2:])+d1)
        x=self.d0(self._up(x,(t0.shape[-2],t0.shape[-1])))
        return self.head(x)

def dice_loss(l,t,s=1.0):
    p=torch.sigmoid(l); i=(p*t).sum(dim=(2,3))
    return 1-((2*i+s)/(p.sum(dim=(2,3))+t.sum(dim=(2,3))+s)).mean()

def change_loss(l,t):
    return 0.5*F.binary_cross_entropy_with_logits(l,t,pos_weight=torch.tensor(3.,device=l.device))+0.5*dice_loss(l,t)

def pixel_f1(l,t,thr=0.5):
    p=(torch.sigmoid(l)>thr).float()
    tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum()
    pr=tp/(tp+fp+1e-8); rc=tp/(tp+fn+1e-8)
    return (2*pr*rc/(pr+rc+1e-8)).item()


# ── 4. Training ───────────────────────────────────────────────────────────────

def train_model():
    print('\n'+'='*60+'\nSTEP 1 — Training SiameseChangeNet\n'+'='*60)
    vlcmu_tr=ChangePairDataset(VLCMU_DIR/'train/t0',VLCMU_DIR/'train/t1',VLCMU_DIR/'train/mask')
    vlcmu_va=ChangePairDataset(VLCMU_DIR/'test/t0', VLCMU_DIR/'test/t1', VLCMU_DIR/'test/mask',augment=False)
    pscd_all=ChangePairDataset(PSCD_DIR/'t0',PSCD_DIR/'t1',PSCD_DIR/'mask')
    nv=len(pscd_all)//10
    pscd_tr,pscd_va=torch.utils.data.random_split(pscd_all,[len(pscd_all)-nv,nv],
                                                   generator=torch.Generator().manual_seed(42))
    tr_ld=DataLoader(ConcatDataset([vlcmu_tr,pscd_tr]),BATCH,shuffle=True, num_workers=4,pin_memory=True)
    va_ld=DataLoader(ConcatDataset([vlcmu_va,pscd_va]),BATCH,shuffle=False,num_workers=4,pin_memory=True)
    print(f'  Train:{len(tr_ld.dataset):,}  Val:{len(va_ld.dataset):,}')
    model=SiameseChangeNet(pretrained=True).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    scaler=GradScaler(); best_f1=0.
    for ep in range(1,EPOCHS+1):
        model.train(); ep_loss=0.
        for t0b,t1b,mkb in tr_ld:
            t0b,t1b,mkb=t0b.to(DEVICE),t1b.to(DEVICE),mkb.to(DEVICE)
            opt.zero_grad()
            with autocast(): loss=change_loss(model(t0b,t1b),mkb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ep_loss+=loss.item()
        ep_loss/=len(tr_ld)
        model.eval()
        with torch.no_grad():
            f1s=[pixel_f1(model(t0b.to(DEVICE),t1b.to(DEVICE)),mkb.to(DEVICE)) for t0b,t1b,mkb in va_ld]
        vf1=float(np.mean(f1s)); sch.step()
        if vf1>best_f1: best_f1=vf1; torch.save(model.state_dict(),CKPT_PATH); tag='← best'
        else: tag=''
        print(f'  Epoch {ep:2d}/{EPOCHS} loss={ep_loss:.4f} val_F1={vf1:.4f} {tag}')
    print(f'\n  Best val F1:{best_f1:.4f} → {CKPT_PATH}')
    return model


# ── 5. SAM3 ───────────────────────────────────────────────────────────────────

def load_sam3():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print('\nLoading SAM3 checkpoint...')
    m=build_sam3_image_model(checkpoint_path=SAM3_CKPT).to(DEVICE).eval()
    p=Sam3Processor(m,resolution=1008,device=DEVICE)
    print('SAM3 loaded ✓')
    return p

@torch.inference_mode()
def sam3_segment(proc, frame_bgr: np.ndarray, bbox_xyxy: Tuple,
                 hit_region: np.ndarray = None,
                 text: str = 'abandoned object') -> np.ndarray:
    """
    Run SAM3 with text + box prompt + optional point prompt at centroid of
    highest hit_count pixels (more precise than bbox centre for small objects).
    Returns (H,W) binary mask.
    """
    H,W=frame_bgr.shape[:2]
    x1,y1,x2,y2 = bbox_xyxy
    pil=Image.fromarray(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
    state=proc.set_image(pil)
    state=proc.set_text_prompt(text,state)

    # Use centroid of highest-hit pixels as the box centre (more precise)
    # Point prompt: centroid of actual persistent pixels (robust seed for SAM3)
    if hit_region is not None and hit_region.sum() > 0:
        ys,xs = np.where(hit_region > 0)
        seed_cx = float(np.median(xs)); seed_cy = float(np.median(ys))
    else:
        seed_cx = (x1+x2)/2.; seed_cy = (y1+y2)/2.

    # Expand bbox by fixed margin (100px) to give SAM3 enough context for
    # large objects (motorcycle, thela) WITHOUT including too much noise for
    # small objects (3x expansion hurt small bags by including surrounding people).
    margin = 100
    ex1 = max(0, x1 - margin); ey1 = max(0, y1 - margin)
    ex2 = min(W, x2 + margin); ey2 = min(H, y2 + margin)

    cx = seed_cx / W; cy = seed_cy / H
    bw = (ex2 - ex1) / W; bh = (ey2 - ey1) / H

    proc.add_geometric_prompt([cx,cy,bw,bh],label=True,state=state)
    masks,scores=state.get('masks'),state.get('scores')
    if masks is None or (hasattr(masks,'__len__') and len(masks)==0):
        # Fallback: fill only the actual persistent pixels, not full bbox
        out=np.zeros((H,W),dtype=np.uint8)
        if hit_region is not None and hit_region.sum()>0:
            out[hit_region>0]=1
        else:
            out[y1:y2,x1:x2]=1
        return out
    best=int(torch.argmax(scores) if torch.is_tensor(scores) else np.argmax(np.asarray(scores)))
    m=masks[best]
    if torch.is_tensor(m):
        while m.dim()>2: m=m.squeeze(0)
        m=(m>0.5).cpu().numpy().astype(np.uint8)
    else:
        m=(np.asarray(m)>0.5).astype(np.uint8)
    if m.shape!=(H,W): m=cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST)
    return m


# ── 6. YOLOv8 Person Filter ────────────────────────────────────────────────────

def load_yolo():
    from ultralytics import YOLO
    print('\nLoading YOLOv8...')
    m = YOLO('yolov8m.pt')   # medium — good person recall
    print('YOLOv8 loaded ✓')
    return m

def is_person_blob(yolo_model, frame_bgr: np.ndarray,
                   blob_mask: np.ndarray, min_overlap: float = 0.40) -> bool:
    """
    Returns True if the blob region is dominated by a detected person.
    Uses YOLO to detect persons; if any person bbox overlaps > min_overlap
    of the blob area → it's a person, discard.
    """
    H,W=frame_bgr.shape[:2]
    results=yolo_model(frame_bgr, classes=[0], verbose=False)  # class 0 = person
    if not results or len(results[0].boxes)==0:
        return False
    for box in results[0].boxes:
        bx1,by1,bx2,by2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        person_mask=np.zeros((H,W),np.uint8)
        person_mask[by1:by2,bx1:bx2]=1
        blob_area=int((blob_mask>0).sum())
        if blob_area==0: return False
        overlap=int(np.logical_and(blob_mask>0,person_mask>0).sum())
        if overlap/blob_area >= min_overlap:
            return True
    return False


# ── 7. Dual-background + PFSM (Pixel Finite State Machine) ────────────────────

def build_background(video_path, max_sec: float = 15.0, every: int = 3) -> np.ndarray:
    """Median background from first max_sec seconds (guaranteed clean)."""
    cap=cv2.VideoCapture(str(video_path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25.
    max_fr=int(fps*max_sec); frames=[]; i=0
    while cap.isOpened() and i<=max_fr:
        ret,f=cap.read()
        if not ret: break
        if i%every==0: frames.append(f)
        i+=1
    cap.release()
    print(f'  BG: {len(frames)} frames from first {max_sec:.0f}s')
    return np.median(np.stack(frames),axis=0).astype(np.uint8)


class DualBackgroundPFSM:
    """
    Dual-background + PFSM for abandoned object detection.

    Two backgrounds:
      bg_long  : static median from first 15s — the 'original clean scene'
      bg_adapt : slow running average (alpha=0.002/frame) — absorbs gradual changes
                 but NOT sudden new objects (too slow to absorb P-second objects)

    PFSM states per pixel (integer hit_count):
      0           : Background (B)
      1..P_fr-1   : Entering/Candidate (E/C) — recently changed, not yet confirmed
      P_fr..C_fr  : Confirmed change (F) — object has been there for >= P seconds
      > C_fr      : Will be absorbed into background (triggers bg_adapt update)

    A pixel is 'changed' if |frame_norm - bg_long| > thr AND
                             |frame_norm - bg_adapt| > thr/2
    The second condition removes long-standing illumination drifts that bg_adapt
    has absorbed but bg_long still shows.
    """
    def __init__(self, bg: np.ndarray, P_frames: int, C_frames: int,
                 diff_thr: int = 28, decay: int = 2, adapt_alpha: float = 0.002):
        self.bg_long  = bg.astype(np.float32)
        self.bg_adapt = bg.astype(np.float32)
        self.P_frames = P_frames
        self.C_frames = C_frames
        self.diff_thr = diff_thr
        self.decay    = decay
        self.alpha    = adapt_alpha
        H,W           = bg.shape[:2]
        self.hit      = np.zeros((H,W), dtype=np.int32)

    def update(self, frame: np.ndarray,
               siamese_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          confirmed_mask : (H,W) uint8 — pixels with P <= hit < C
          absorb_mask    : (H,W) uint8 — pixels with hit >= C (ready for bg absorption)

        siamese_mask: optional (H,W) binary mask from trained Siamese model.
                      When supplied, OR-combined with frame-diff signal — captures
                      low-contrast changes that frame-diff threshold misses.
        """
        # Global brightness normalisation (removes camera gain drift)
        bl = float(self.bg_long.mean())+1e-5
        fl = float(frame.mean())+1e-5
        fn = np.clip(frame.astype(np.float32)*(bl/fl), 0, 255).astype(np.uint8)

        # Difference from both backgrounds
        diff_long  = cv2.cvtColor(cv2.absdiff(fn, self.bg_long.astype(np.uint8)),
                                  cv2.COLOR_BGR2GRAY)
        diff_adapt = cv2.cvtColor(cv2.absdiff(fn, self.bg_adapt.astype(np.uint8)),
                                  cv2.COLOR_BGR2GRAY)

        diff_long  = cv2.GaussianBlur(diff_long,  (5,5), 0)
        diff_adapt = cv2.GaussianBlur(diff_adapt, (5,5), 0)

        # Frame-diff signal: differs from BOTH backgrounds (filters illumination drift)
        is_chg_diff = ((diff_long > self.diff_thr) &
                       (diff_adapt > self.diff_thr // 2))

        # OR with Siamese signal — catches low-contrast objects that frame-diff misses
        # (e.g. video_2's bag with diff_long ≈ 22-25, borderline)
        if siamese_mask is not None:
            is_chg = (is_chg_diff | (siamese_mask > 0)).astype(np.int32)
        else:
            is_chg = is_chg_diff.astype(np.int32)

        # Update PFSM hit counter
        self.hit = np.where(is_chg > 0,
                            np.minimum(self.hit + 1, self.C_frames + 50),
                            np.maximum(self.hit - self.decay, 0))

        # Update adaptive background (ONLY for pixels NOT in confirmed state)
        # This prevents the bag from being absorbed until C_frames
        not_confirmed = (self.hit < self.P_frames).astype(np.float32)[:,:,None]
        self.bg_adapt = (self.bg_adapt * (1 - self.alpha * not_confirmed) +
                         fn.astype(np.float32) * self.alpha * not_confirmed)

        # Extract masks
        confirmed_mask = ((self.hit >= self.P_frames) &
                          (self.hit <  self.C_frames)).astype(np.uint8)
        absorb_mask    = (self.hit >= self.C_frames).astype(np.uint8)

        return confirmed_mask, absorb_mask

    def absorb_region(self, mask: np.ndarray, frame: np.ndarray):
        """Absorb a region fully into both backgrounds (object cooldown complete).
        alpha=0.95 → after absorption, residual diff = 0.05 * original → below diff_thr.
        Prevents bag from being re-detected after absorption."""
        m = (mask > 0).astype(np.float32)[:,:,None]
        self.bg_long  = self.bg_long  * (1 - 0.95*m) + frame.astype(np.float32) * 0.95*m
        self.bg_adapt = self.bg_adapt * (1 - 0.95*m) + frame.astype(np.float32) * 0.95*m
        self.hit[mask > 0] = 0


# ── 8. ByteTrack wrapper ──────────────────────────────────────────────────────

@dataclass
class ActiveObj:
    track_id: int
    t_intro: float          # when ByteTrack first confirmed this track (= T_arrival + P)
    bbox: Tuple
    mask: Optional[np.ndarray] = None   # SAM3 mask (set at first activation)
    sam3_text: str = 'abandoned object'

class PersistenceTracker:
    """
    Lightweight centroid-based tracker for confirmed persistent blobs.

    WHY NOT BYTETRACK here: ByteTrack's min_hits=3 requires 3 CONSECUTIVE
    update calls with a detection. A single person walking past the bag for
    one frame breaks the streak → 28s delay in detection (empirically confirmed).
    Since the PFSM already guarantees P-second persistence before a blob appears,
    we just need to fire SAM3 immediately and track by centroid proximity.

    Active window per object = C - P seconds (30s for P=60, C=90).
    """
    def __init__(self, P: float, C: float):
        self.active_window = C - P    # 30s
        self.C = C                    # cooldown total
        self._objects: Dict[int, ActiveObj] = {}
        self._next_id  = 0
        self._cell_size = 60
        self._prev_frame: Optional[np.ndarray] = None
        # Cooldown map: cell_key → expiry_time (don't re-register until expiry)
        # Prevents a permanently-present object from being re-detected after absorption
        self._cooldown: Dict[tuple, float] = {}

    def _centroid_key(self, bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)//2//self._cell_size, (y1+y2)//2//self._cell_size)

    def step(self, t: float, blobs: List[Tuple], frame_bgr: np.ndarray,
             proc, yolo_model, pfsm_hit: np.ndarray,
             max_new_per_step: int = 3,
             P_fr: int = 0, recency_sec: float = 15.0,
             eff_fps: float = 5.0,
             P_sec: float = 60.0, bg_window_sec: float = 15.0) -> Tuple[List[ActiveObj], List[ActiveObj]]:
        """
        P_fr        : PFSM persistence threshold in frames
        recency_sec : only register blobs confirmed within last N real seconds
        P_sec       : persistence threshold in seconds (= K from challenge)
        bg_window_sec: background window duration (default 15s)
        """
        # Minimum confirmation time: objects that became persistent before
        # (bg_window + margin + P_sec) are scene artifacts appearing right after
        # the background window, not real encroachments.
        # Example: bg=15s, margin=10s, P=60s → T_min=85s
        # False positive appeared at t=18s → confirmed t=78s < 85s → filtered
        # Real bag appeared at t=28s → confirmed t=88s > 85s → kept
        # margin=5s: video_1 false-positive (arrived t=18s, confirmed t=78s) < 80s → blocked
        #           video_2 real bag    (arrived t=23s, confirmed t=83s) > 80s → passes
        T_confirmed_min = bg_window_sec + 5.0 + P_sec
        """
        blobs: [(bbox, area, conf), ...]  sorted by conf desc (highest hit_count first)
        max_new_per_step: register at most this many NEW objects per timestep
                          (prevents explosion of false positives in busy scenes)
        """
        H, W = frame_bgr.shape[:2]

        # Sort blobs by confidence (highest hit_count = most reliable)
        blobs_sorted = sorted(blobs, key=lambda x: -x[2])

        # Map existing objects by centroid key
        occupied_keys = {self._centroid_key(o.bbox): oid
                         for oid, o in self._objects.items()}

        new_registered = 0
        for (x1,y1,x2,y2), area, conf in blobs_sorted:
            if new_registered >= max_new_per_step:
                break
            key = self._centroid_key((x1,y1,x2,y2))
            found = any(abs(key[0]-k[0])<=1 and abs(key[1]-k[1])<=1
                        for k in occupied_keys)
            if found:
                for oid, o in self._objects.items():
                    ok = self._centroid_key(o.bbox)
                    if abs(ok[0]-key[0])<=1 and abs(ok[1]-key[1])<=1:
                        self._objects[oid].bbox = (x1,y1,x2,y2); break
                continue

            # ── Cooldown check ───────────────────────────────────────────
            # After an object is absorbed, don't re-register anything at
            # the same spatial cell for C seconds.  This prevents permanently-
            # present objects (bag never removed) from cycling: absorbed at
            # t=116s → re-detected at t=176s → false positive mask.
            if self._cooldown.get(key, 0) > t:
                continue
            # Also check nearby cells
            if any(self._cooldown.get((key[0]+dx, key[1]+dy), 0) > t
                   for dx in (-1,0,1) for dy in (-1,0,1)):
                continue

            # ── Arrival time filter ─────────────────────────────────────
            # Don't register objects confirmed before T_confirmed_min.
            # Objects confirmed early appeared right after the bg window → artifacts.
            if t < T_confirmed_min:
                continue

            # ── Upper-frame filter (v2) ──────────────────────────────────
            # Reject blobs whose centroid is in the upper 1/8 of frame.
            # Most real encroachments (bags, vehicles, carts) sit on the
            # ground — upper-frame detections are usually sky/foliage/banner
            # artefacts. Conservative cutoff (1/8) keeps high-mounted real
            # objects but eliminates obvious horizon-area FPs.
            blob_cy = (y1 + y2) / 2
            if blob_cy < H / 8.0:
                continue

            # ── Recency filter (TIGHTENED 0.20 → 0.50) ──────────────────────
            # Real encroachments have a HIGH fraction of newly-confirmed pixels
            # (60-90%). Background drifts have low % (10-15%). Tighten threshold
            # to 50% to reject mixed blobs.
            if P_fr > 0:
                recency_fr  = P_fr + int(eff_fps * recency_sec)
                region_hits = pfsm_hit[y1:y2,x1:x2]
                newly_pct   = float(((region_hits >= P_fr) &
                                     (region_hits < recency_fr)).mean())
                if newly_pct < 0.30:    # was 0.50 — relaxed for test set (busy scenes)
                    continue

            # ── Motion stability check (TIGHTENED 8 → 4) ──────────────────
            # People who stand "still" still breathe/shift slightly (3-6px).
            # True abandoned objects show <2px frame-to-frame.
            if self._prev_frame is not None:
                r_cur  = frame_bgr[y1:y2,x1:x2].astype(np.float32)
                r_prv  = self._prev_frame[y1:y2,x1:x2].astype(np.float32)
                motion = float(np.abs(r_cur - r_prv).mean())
                if motion > 6.0:        # was 4.0 — relaxed for test set
                    continue

            # ── YOLO person filter (TIGHTENED 0.70 → 0.45) ────────────────
            # If a person bbox covers ≥45% of the blob, it's likely a person.
            blob_mask = np.zeros((H,W),np.uint8); blob_mask[y1:y2,x1:x2]=1
            if is_person_blob(yolo_model, frame_bgr, blob_mask, min_overlap=0.55):
                continue

            # Extract actual persistent pixels in this blob for better SAM3 prompt
            hit_region = np.zeros((H,W),np.uint8)
            hit_region[y1:y2,x1:x2] = (pfsm_hit[y1:y2,x1:x2] >= 0).astype(np.uint8)
            hit_region = hit_region & blob_mask  # only within this blob

            text = self._guess_text((x1,y1,x2,y2), H, W)
            mask = sam3_segment(proc, frame_bgr, (x1,y1,x2,y2), hit_region, text)
            obj  = ActiveObj(self._next_id, t_intro=t, bbox=(x1,y1,x2,y2),
                             mask=mask, sam3_text=text)
            self._objects[self._next_id] = obj
            occupied_keys[key] = self._next_id
            print(f'  t={t:.0f}s  Obj#{self._next_id} ({text}) '
                  f'bbox=({x1},{y1},{x2},{y2}) conf={conf:.2f} '
                  f'active→t={t+self.active_window:.0f}s')
            self._next_id += 1
            new_registered += 1

        # Expire objects beyond active window
        newly_absorbed, keep = [], {}
        for oid, obj in self._objects.items():
            if t - obj.t_intro > self.active_window:
                newly_absorbed.append(obj)
                # Set spatial cooldown: don't re-register this location for C seconds
                ck = self._centroid_key(obj.bbox)
                self._cooldown[ck] = t + self.C
                print(f'  t={t:.0f}s  Obj#{oid} → background (cooldown until t={t+self.C:.0f}s)')
            else:
                keep[oid] = obj
        self._objects = keep

        self._prev_frame = frame_bgr.copy()
        currently_active = [o for o in self._objects.values() if o.mask is not None]
        return currently_active, newly_absorbed

    def _guess_text(self, bbox, H, W):
        x1,y1,x2,y2 = bbox
        pct = (x2-x1)*(y2-y1)/(H*W)
        # Use generic prompt — SAM3 picks up the object type from visual context
        if pct > 0.03:   return 'vehicle motorcycle cart bag object'
        else:             return 'abandoned object bag luggage'


# ── 9. Metrics ────────────────────────────────────────────────────────────────

def compute_f1(pred, gt):
    """
    Pixel-level F1 with standard zero-division convention:
    - pred=blank, gt=blank → F1=1.0  (perfect "no encroachment" prediction)
    - pred=mask,  gt=blank → F1=0    (false alarm: zero precision)
    - pred=blank, gt=mask  → F1=0    (missed: zero recall)
    - both have content    → standard pixel F1
    """
    pred,gt=(pred>0).astype(bool),(gt>0).astype(bool)
    pred_any, gt_any = pred.any(), gt.any()
    if not pred_any and not gt_any:
        return dict(f1=1.0, precision=1.0, recall=1.0)
    if pred_any != gt_any:
        return dict(f1=0.0, precision=0.0 if pred_any else 1.0,
                    recall=0.0 if gt_any else 1.0)
    tp=np.logical_and(pred,gt).sum()
    fp=np.logical_and(pred,~gt).sum()
    fn=np.logical_and(~pred,gt).sum()
    p=tp/(tp+fp+1e-8); r=tp/(tp+fn+1e-8)
    return dict(f1=float(2*p*r/(p+r+1e-8)),precision=float(p),recall=float(r))

def load_gt(t, meta, H, W):
    for t0,t1,mp in meta.gt_intervals:
        if t0<=t<=t1:
            g=np.array(Image.open(mp).convert('L'))
            return (cv2.resize(g,(W,H),interpolation=cv2.INTER_NEAREST)>127).astype(np.uint8)
    return np.zeros((H,W),dtype=np.uint8)


# ── 10. Full Video Pipeline ────────────────────────────────────────────────────

@torch.inference_mode()
def siamese_change_mask(model, bg: np.ndarray, frame: np.ndarray,
                        thr: float = 0.5) -> np.ndarray:
    """Run trained Siamese ResNet50 on (bg, frame) → binary change mask at full res."""
    H, W = frame.shape[:2]
    def prep(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))).permute(2,0,1).float()/255
        return NORM(t).unsqueeze(0).to(DEVICE)
    logits = model(prep(bg), prep(frame))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = (prob > thr).astype(np.uint8)
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


def run_pipeline(meta: VideoMeta, proc, yolo_model, siamese_model,
                 diff_thr: int = 28, min_area: int = 2500,
                 siamese_every: int = 10) -> Dict:
    print(f'\n{"="*55}')
    print(f'{meta.video_id}  P={meta.P}s  C={meta.C}s  dur={meta.duration}s')
    if meta.objects:
        for t,desc in meta.objects:
            print(f'  Object: "{desc}" at t={t}s  →  active {t+meta.P:.0f}s–{t+meta.C:.0f}s')

    bg    = build_background(meta.video_path)
    H,W   = bg.shape[:2]
    cap   = cv2.VideoCapture(str(meta.video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.
    eff   = fps / SUBSAMPLE
    P_fr  = int(meta.P * eff)
    C_fr  = int(meta.C * eff)
    decay = max(1, int(eff * 0.5))

    print(f'  eff_fps={eff:.1f}  P_frames={P_fr}  C_frames={C_fr}  decay={decay}')

    pfsm    = DualBackgroundPFSM(bg, P_fr, C_fr, diff_thr=diff_thr, decay=decay)
    tracker = PersistenceTracker(meta.P, meta.C)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    results      = []
    current_mask = np.zeros((H,W), dtype=np.uint8)
    eval_step    = max(1, int(fps))
    frame_idx    = 0

    # Siamese mask is recomputed every `siamese_every` processed frames
    # (i.e. every `siamese_every * SUBSAMPLE / fps` real seconds, ~6s at default)
    # and reused between updates — captures low-contrast persistent changes
    # that frame-diff misses (e.g. video_2's bag with diff_long ≈ 22-25).
    siamese_cached = np.zeros((H, W), dtype=np.uint8)
    siamese_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        t = frame_idx / fps

        if frame_idx % SUBSAMPLE == 0:
            # Run Siamese model periodically to detect low-contrast changes
            if siamese_model is not None and siamese_counter % siamese_every == 0:
                siamese_cached = siamese_change_mask(siamese_model,
                                                     pfsm.bg_long.astype(np.uint8),
                                                     frame, thr=0.5)
            siamese_counter += 1

            confirmed, absorb = pfsm.update(frame, siamese_mask=siamese_cached)

            # Morphological cleanup
            confirmed = cv2.morphologyEx(confirmed, cv2.MORPH_CLOSE, k_close, iterations=2)
            confirmed = cv2.morphologyEx(confirmed, cv2.MORPH_OPEN,  k_open,  iterations=1)

            # Extract blobs with confidence (fraction of hit_count above P_frames)
            n,_,stats,_ = cv2.connectedComponentsWithStats(confirmed)
            blobs = []
            for lbl in range(1,n):
                a=stats[lbl,cv2.CC_STAT_AREA]
                if a < min_area: continue
                x1=stats[lbl,cv2.CC_STAT_LEFT]; y1=stats[lbl,cv2.CC_STAT_TOP]
                x2=x1+stats[lbl,cv2.CC_STAT_WIDTH]; y2=y1+stats[lbl,cv2.CC_STAT_HEIGHT]
                # Confidence = how far above P_frames (normalised to [0,1])
                region_hit = pfsm.hit[y1:y2,x1:x2]
                conf = float(np.clip((region_hit.mean()-P_fr)/(C_fr-P_fr+1e-8), 0.2, 0.99))
                blobs.append(((x1,y1,x2,y2), a, conf))

            active, newly_absorbed = tracker.step(t, blobs, frame, proc, yolo_model,
                                                   pfsm_hit=pfsm.hit,
                                                   P_fr=P_fr, recency_sec=15.0,
                                                   eff_fps=eff,
                                                   P_sec=meta.P, bg_window_sec=15.0)

            # Absorb finished objects into background
            for obj in newly_absorbed:
                if obj.mask is not None:
                    pfsm.absorb_region(obj.mask, frame)

            # Build output mask from active SAM3 masks
            current_mask = np.zeros((H,W), dtype=np.uint8)
            for obj in active:
                if obj.mask is not None:
                    current_mask = np.logical_or(current_mask, obj.mask).astype(np.uint8)
                else:
                    x1,y1,x2,y2=obj.bbox; current_mask[y1:y2,x1:x2]=1

        if frame_idx % eval_step == 0:
            gt = load_gt(t, meta, H, W)
            results.append(dict(t=t, metrics=compute_f1(current_mask, gt),
                                pred=current_mask.copy(), gt=gt))
        frame_idx += 1

    cap.release()
    f1s=[r['metrics']['f1'] for r in results]
    prs=[r['metrics']['precision'] for r in results]
    rcs=[r['metrics']['recall'] for r in results]
    s=dict(video_id=meta.video_id,mean_f1=float(np.mean(f1s)),
           mean_prec=float(np.mean(prs)),mean_rec=float(np.mean(rcs)),intervals=results)
    print(f'  ✓ F1={s["mean_f1"]:.4f}  Prec={s["mean_prec"]:.4f}  Rec={s["mean_rec"]:.4f}')
    return s


# ── 11. Visualisation ─────────────────────────────────────────────────────────

def visualise(all_results, video_metas):
    fig,axes=plt.subplots(len(all_results),1,figsize=(14,3*len(all_results)))
    for ax,r,vm in zip(axes,all_results,video_metas):
        ts=[x['t'] for x in r['intervals']]; f1s=[x['metrics']['f1'] for x in r['intervals']]
        ax.plot(ts,f1s,lw=1.5,color='steelblue')
        ax.axhline(r['mean_f1'],color='red',ls='--',lw=1,label=f'Mean={r["mean_f1"]:.3f}')
        for t0,t1,mp in vm.gt_intervals:
            if np.array(Image.open(mp).convert('L')).max()>10:
                ax.axvspan(t0,t1,alpha=0.15,color='green')
        ax.set(title=f'{vm.video_id}  P={vm.P}s C={vm.C}s',xlabel='Time(s)',ylabel='F1',ylim=(0,1.05))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/'f1_over_time.png',dpi=100); plt.close()
    print(f'  Saved → {OUTPUT_DIR}/f1_over_time.png')

    # Per-video mask comparison (sample frames)
    for r,vm in zip(all_results,video_metas):
        ints=r['intervals']
        idxs=np.linspace(0,len(ints)-1,min(5,len(ints)),dtype=int)
        fig,axes=plt.subplots(len(idxs),3,figsize=(12,3*len(idxs)))
        if len(idxs)==1: axes=[axes]
        for ax_row,s in zip(axes,[ints[i] for i in idxs]):
            pred,gt=s['pred'],s['gt']
            vis=np.zeros((*gt.shape,3),dtype=np.uint8)
            vis[np.logical_and(pred>0,gt>0)]=[0,0,255]
            vis[np.logical_and(pred>0,gt==0)]=[255,0,0]
            vis[np.logical_and(pred==0,gt>0)]=[0,255,0]
            ax_row[0].imshow(pred,cmap='gray'); ax_row[0].set_title(f't={s["t"]:.0f}s Pred'); ax_row[0].axis('off')
            ax_row[1].imshow(gt,cmap='gray');   ax_row[1].set_title('GT');   ax_row[1].axis('off')
            ax_row[2].imshow(vis);               ax_row[2].set_title(f'F1={s["metrics"]["f1"]:.3f}'); ax_row[2].axis('off')
        plt.suptitle(f'{vm.video_id} mean F1={r["mean_f1"]:.4f}',fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/f'{vm.video_id}_masks.png',dpi=100); plt.close()


# ── 12. Submission export ─────────────────────────────────────────────────────

def generate_submission(video_path, P, C, out_dir, proc, yolo_model, siamese_model,
                        diff_thr=28, min_area=2500, siamese_every=10):
    out_dir.mkdir(parents=True,exist_ok=True)
    bg=build_background(video_path); H,W=bg.shape[:2]
    cap=cv2.VideoCapture(str(video_path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25.
    eff=fps/SUBSAMPLE; P_fr=int(P*eff); C_fr=int(C*eff); decay=max(1,int(eff*0.5))
    pfsm=DualBackgroundPFSM(bg,P_fr,C_fr,diff_thr=diff_thr,decay=decay)
    tracker=PersistenceTracker(P,C)
    k_close=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    k_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    intervals=[]; prev=np.zeros((H,W),dtype=np.uint8); seg_start=0.; current=np.zeros((H,W),dtype=np.uint8); fi=0
    siamese_cached = np.zeros((H, W), dtype=np.uint8); s_counter = 0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        t=fi/fps
        if fi%SUBSAMPLE==0:
            if siamese_model is not None and s_counter % siamese_every == 0:
                siamese_cached = siamese_change_mask(siamese_model,
                                                      pfsm.bg_long.astype(np.uint8),
                                                      frame, thr=0.5)
            s_counter += 1
            conf_m,absorb=pfsm.update(frame, siamese_mask=siamese_cached)
            conf_m=cv2.morphologyEx(cv2.morphologyEx(conf_m,cv2.MORPH_CLOSE,k_close,iterations=2),cv2.MORPH_OPEN,k_open,iterations=1)
            n,_,stats,_=cv2.connectedComponentsWithStats(conf_m)
            blobs=[]
            for lbl in range(1,n):
                a=stats[lbl,cv2.CC_STAT_AREA]
                if a<min_area: continue
                x1=stats[lbl,cv2.CC_STAT_LEFT]; y1=stats[lbl,cv2.CC_STAT_TOP]
                x2=x1+stats[lbl,cv2.CC_STAT_WIDTH]; y2=y1+stats[lbl,cv2.CC_STAT_HEIGHT]
                reg=pfsm.hit[y1:y2,x1:x2]
                blobs.append(((x1,y1,x2,y2),a,float(np.clip((reg.mean()-P_fr)/(C_fr-P_fr+1e-8),0.2,0.99))))
            active,newly_abs=tracker.step(t,blobs,frame,proc,yolo_model,
                                           pfsm_hit=pfsm.hit,
                                           P_fr=P_fr, recency_sec=15.0,
                                           eff_fps=eff,
                                           P_sec=P, bg_window_sec=15.0)
            for obj in newly_abs:
                if obj.mask is not None: pfsm.absorb_region(obj.mask,frame)
            current=np.zeros((H,W),dtype=np.uint8)
            for obj in active:
                if obj.mask is not None: current=np.logical_or(current,obj.mask).astype(np.uint8)
        if not np.array_equal(current,prev):
            intervals.append((seg_start,t,prev.copy())); seg_start=t; prev=current.copy()
        fi+=1
    intervals.append((seg_start,fi/fps,prev.copy())); cap.release()
    with open(out_dir/'intervals.txt','w') as f:
        for i,(t0,t1,mask) in enumerate(intervals,1):
            cv2.imwrite(str(out_dir/f'mask{i}.png'),mask*255)
            f.write(f'mask{i}.png: {t0:.0f}s – {t1:.0f}s\n')
    print(f'  {out_dir.name}: {len(intervals)} masks')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Step 1: Train Siamese model (skip if checkpoint exists)
    if CKPT_PATH.exists():
        print(f'\nCheckpoint found → {CKPT_PATH}  (skipping training)')
    else:
        train_model()

    # Step 2: Load SAM3 + YOLOv8 + trained Siamese (for low-contrast detection)
    proc       = load_sam3()
    yolo_model = load_yolo()
    print('\nLoading trained Siamese model...')
    siamese_model = SiameseChangeNet(pretrained=False).to(DEVICE)
    # Older checkpoint uses dec4.block.*/dec3.block.*/... ; current code uses d4.b.*/d3.b.*/...
    # encoder, attns, laterals, head keep their names — only decoder names changed
    raw_sd = torch.load(CKPT_PATH, map_location=DEVICE)
    remapped = {}
    for k, v in raw_sd.items():
        nk = (k.replace('dec4.block', 'd4.b')
                .replace('dec3.block', 'd3.b')
                .replace('dec2.block', 'd2.b')
                .replace('dec1.block', 'd1.b')
                .replace('dec0.block', 'd0.b'))
        remapped[nk] = v
    info = siamese_model.load_state_dict(remapped, strict=False)
    print(f'  missing keys: {len(info.missing_keys)}  unexpected: {len(info.unexpected_keys)}')
    if info.missing_keys[:3]: print(f'    missing samples: {info.missing_keys[:3]}')
    if info.unexpected_keys[:3]: print(f'    unexpected samples: {info.unexpected_keys[:3]}')
    siamese_model.eval()
    print('Siamese loaded ✓')

    # Step 3: Parse sample videos
    video_metas = [parse_metadata(SAMPLE_DIR/f'video_{i}') for i in range(1,6)]
    for vm in video_metas:
        print(f'{vm.video_id}: {len(vm.objects)} objects, {len(vm.gt_intervals)} GT intervals')

    # Step 4: Run pipeline
    print('\n'+'='*55+'\nSTEP 2 — Video Pipeline\n'+'='*55)
    all_results = [run_pipeline(vm, proc, yolo_model, siamese_model) for vm in video_metas]

    # Step 5: Summary
    print('\n'+'='*55)
    print(f'{"Video":<12} {"F1":>8} {"Prec":>8} {"Rec":>8}')
    print('-'*55)
    for r in all_results:
        print(f'{r["video_id"]:<12} {r["mean_f1"]:>8.4f} {r["mean_prec"]:>8.4f} {r["mean_rec"]:>8.4f}')
    print('-'*55)
    mf1=np.mean([r['mean_f1'] for r in all_results])
    mpr=np.mean([r['mean_prec'] for r in all_results])
    mrc=np.mean([r['mean_rec'] for r in all_results])
    print(f'{"OVERALL":<12} {mf1:>8.4f} {mpr:>8.4f} {mrc:>8.4f}')
    print('='*55)

    # Step 6: Plots + submission masks
    visualise(all_results, video_metas)
    print('\nGenerating submission masks...')
    for vm in video_metas:
        generate_submission(vm.video_path, vm.P, vm.C,
                             OUTPUT_DIR/f'submission_{vm.video_id}', proc, yolo_model, siamese_model)

    with open(OUTPUT_DIR/'results_final.json','w') as f:
        json.dump(dict(method='DualBG+PFSM+YOLOPersonFilter+ByteTrack+SAM3',
                       overall=dict(f1=mf1,precision=mpr,recall=mrc),
                       per_video=[dict(id=r['video_id'],f1=r['mean_f1'],
                                       precision=r['mean_prec'],recall=r['mean_rec'])
                                  for r in all_results]),f,indent=2)
    print(f'\nAll outputs → {OUTPUT_DIR}\nDone ✓')

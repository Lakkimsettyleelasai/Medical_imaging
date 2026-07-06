"""
Pituitary Adenoma 2D Segmentation — V2 Training (Resume from checkpoint)
=========================================================================
Resumes from best_model.pth, trains to 500 epochs, saves to best_model_v2.pth
"""

import os, json, random, warnings
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from monai.networks.nets import BasicUNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

# =====================================================================
# CONFIG
# =====================================================================
CFG = {
    "dataset_root": "/home/23bq1a4293/Personal/project/seg_data/pitutary/pituitary_tumor_dataset",
    "seg_dir":      "derivatives/segmentations",
    "output_dir":   "/home/23bq1a4293/Personal/project/models/Segmentation/pituitary",

    "target_size":  (512, 512),
    "num_classes":  6,
    "val_patients": 10,
    "seed":         42,

    "epochs":       500,      # extended
    "batch_size":   8,
    "lr":           1e-3,
    "weight_decay": 1e-5,
    "num_workers":  4,

    "features": (32, 32, 64, 128, 256, 32),

    # bg, tumor, gland, ICA-L, ICA-R, chiasm
    "class_weights": [0.1, 1.0, 2.5, 3.0, 3.0, 3.5],

    "aug_prob": 0.5,
    "patience": 100,          # extended
}

# ── RESUME SETTINGS ───────────────────────────────────────────────────
RESUME_FROM = "/home/23bq1a4293/Personal/project/models/Segmentation/pituitary/best_model.pth"
SAVE_AS     = "best_model_v2.pth"   # won't overwrite v1

set_determinism(seed=CFG["seed"])
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

os.makedirs(CFG["output_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
print(f"Output  : {CFG['output_dir']}")

# =====================================================================
# LOAD CHECKPOINT
# =====================================================================
print(f"\nLoading checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location="cpu", weights_only=False)
print(f"✅ Resuming from epoch {ckpt['epoch']}, best Dice={ckpt['best_dice']:.4f}")

# =====================================================================
# STEP 1 — COLLECT CASES
# =====================================================================
dataset_root = Path(CFG["dataset_root"])
seg_root     = dataset_root / CFG["seg_dir"]

cases, missing = [], []
for sub in sorted(dataset_root.iterdir()):
    if not sub.is_dir() or not sub.name.startswith("sub-"):
        continue
    pid      = sub.name
    ce_path  = sub / "anat" / f"{pid}_acq-CECor_T1w.nii.gz"
    cor_path = sub / "anat" / f"{pid}_acq-Cor_T1w.nii.gz"
    seg_path = seg_root / pid / "anat" / f"{pid}_label-groundTruth.nii.gz"

    if ce_path.exists() and cor_path.exists() and seg_path.exists():
        cases.append({"pid": pid, "ce": str(ce_path),
                      "cor": str(cor_path), "seg": str(seg_path)})
    else:
        missing.append(pid)

print(f"Valid cases : {len(cases)}/50")
if missing:
    print(f"Missing     : {missing}")

# =====================================================================
# STEP 2 — SAME TRAIN/VAL SPLIT AS V1 (load from saved json)
# =====================================================================
split_path = os.path.join(CFG["output_dir"], "train_val_split.json")
with open(split_path) as f:
    split = json.load(f)

train_pids = split["train"]
val_pids   = split["val"]

train_cases = [c for c in cases if c["pid"] in train_pids]
val_cases   = [c for c in cases if c["pid"] in val_pids]

print(f"Train: {len(train_cases)} patients | Val: {len(val_cases)} patients")
print(f"Val PIDs: {val_pids}")

# =====================================================================
# STEP 3 — SLICE EXTRACTION
# =====================================================================
def normalize(arr):
    mean = arr.mean()
    std  = arr.std() + 1e-8
    return ((arr - mean) / std).astype(np.float32)


def load_and_extract_slices(case_list, target_size=(512, 512)):
    slices = []
    for c in case_list:
        ce_arr  = nib.load(c["ce"]).get_fdata(dtype=np.float32)
        cor_arr = nib.load(c["cor"]).get_fdata(dtype=np.float32)
        seg_arr = nib.load(c["seg"]).get_fdata().astype(np.uint8)

        H, D, W = seg_arr.shape
        if H != target_size[0] or W != target_size[1]:
            scale   = (target_size[0]/H, 1.0, target_size[1]/W)
            ce_arr  = zoom(ce_arr,  scale, order=1)
            cor_arr = zoom(cor_arr, scale, order=1)
            seg_arr = zoom(seg_arr, scale, order=0)

        ce_arr  = normalize(ce_arr)
        cor_arr = normalize(cor_arr)

        for y in range(seg_arr.shape[1]):
            image = np.stack([ce_arr[:, y, :], cor_arr[:, y, :]], axis=0)
            slices.append({
                "image":     image,
                "label":     seg_arr[:, y, :].astype(np.int64),
                "pid":       c["pid"],
                "slice_idx": y,
            })
    return slices


print("\nExtracting slices...")
train_slices = load_and_extract_slices(train_cases, CFG["target_size"])
val_slices   = load_and_extract_slices(val_cases,   CFG["target_size"])

train_slices_fg = [s for s in train_slices if s["label"].max() > 0]
print(f"Train slices (all/fg): {len(train_slices)}/{len(train_slices_fg)}")
print(f"Val   slices         : {len(val_slices)}")

# =====================================================================
# STEP 4 — DATASET & AUGMENTATION
# =====================================================================
class SliceDataset(Dataset):
    def __init__(self, slices, augment=False, aug_prob=0.5):
        self.slices   = slices
        self.augment  = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        item  = self.slices[idx]
        image = torch.from_numpy(item["image"].copy()).float()
        label = torch.from_numpy(item["label"].copy()).long()
        if self.augment:
            image, label = self._augment(image, label)
        return image, label

    def _augment(self, image, label):
        p = self.aug_prob

        # Horizontal flip + swap ICA L/R
        if random.random() < p:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
            lf = label.clone()
            lf[label == 3] = 4
            lf[label == 4] = 3
            label = lf

        # Vertical flip
        if random.random() < p * 0.3:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[0])

        # 90-deg rotation
        if random.random() < p * 0.3:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])

        # Brightness / contrast
        if random.random() < p:
            for ch in range(image.shape[0]):
                image[ch] = image[ch] * random.uniform(0.8, 1.2) \
                            + random.uniform(-0.1, 0.1)

        # Gaussian noise
        if random.random() < p * 0.5:
            image = image + torch.randn_like(image) * 0.05

        return image, label


train_ds = SliceDataset(train_slices_fg, augment=True,  aug_prob=CFG["aug_prob"])
val_ds   = SliceDataset(val_slices,      augment=False)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True,
                          drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# =====================================================================
# STEP 5 — MODEL (load weights from checkpoint)
# =====================================================================
model = BasicUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=CFG["num_classes"],
    features=CFG["features"],
    act="PRELU",
    norm="INSTANCE",
    dropout=0.1,
).to(device)

model.load_state_dict(ckpt["model"])
print(f"✅ Model weights loaded from checkpoint")

# =====================================================================
# STEP 6 — LOSS / OPTIMIZER / SCHEDULER (restore states)
# =====================================================================
class_weights   = torch.tensor(CFG["class_weights"], dtype=torch.float32).to(device)

dice_focal_loss = DiceFocalLoss(
    include_background=True, to_onehot_y=True, softmax=True,
    gamma=2.0, weight=class_weights, lambda_dice=0.5, lambda_focal=0.5,
)
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=CFG["lr"], weight_decay=CFG["weight_decay"])

def lr_lambda(epoch):
    warmup = 10
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(CFG["epochs"] - warmup, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Restore optimizer & scheduler states
optimizer.load_state_dict(ckpt["optimizer"])
scheduler.load_state_dict(ckpt["scheduler"])

# Move optimizer state to correct device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

start_epoch = ckpt["epoch"] + 1
best_dice   = ckpt["best_dice"]
best_epoch  = ckpt["epoch"]
print(f"✅ Optimizer/scheduler restored. Starting from epoch {start_epoch}")
print(f"   Current best Dice = {best_dice:.4f} (epoch {best_epoch})")

dice_metric = DiceMetric(include_background=False,
                         reduction="mean_batch", get_not_nans=True)
post_pred  = AsDiscrete(argmax=True,  to_onehot=CFG["num_classes"])
post_label = AsDiscrete(to_onehot=CFG["num_classes"])

# =====================================================================
# STEP 7 — TRAINING LOOP
# =====================================================================
LABEL_NAMES = ["tumor", "gland", "ICA-L", "ICA-R", "chiasm"]

history = {
    "train_loss": [], "val_loss": [], "val_dice_mean": [], "lr": [],
    **{f"val_dice_{n}": [] for n in LABEL_NAMES},
}

no_improve = 0

print("\n" + "="*75)
print(f"TRAINING V2 — epochs {start_epoch} to {CFG['epochs']}")
print("="*75)

for epoch in range(start_epoch, CFG["epochs"] + 1):

    # ── TRAIN ─────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = (dice_focal_loss(logits, labels.unsqueeze(1)) +
                0.3 * ce_loss_fn(logits, labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()
    lr_now = optimizer.param_groups[0]["lr"]

    # ── VALIDATE ──────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    dice_metric.reset()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_loss += (dice_focal_loss(logits, labels.unsqueeze(1)) +
                         0.3 * ce_loss_fn(logits, labels)).item()
            dice_metric(
                y_pred=[post_pred(p) for p in logits],
                y=[post_label(l.unsqueeze(0)) for l in labels],
            )

    val_loss /= len(val_loader)
    dice_vals, _ = dice_metric.aggregate()
    dpc       = dice_vals.cpu().numpy()
    mean_dice = float(np.nanmean(dpc))

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_dice_mean"].append(mean_dice)
    history["lr"].append(lr_now)
    for i, n in enumerate(LABEL_NAMES):
        history[f"val_dice_{n}"].append(float(dpc[i]) if i < len(dpc) else 0.0)

    if epoch % 10 == 0 or epoch == start_epoch:
        print(f"Ep {epoch:03d}/{CFG['epochs']} | LR={lr_now:.2e} | "
              f"TrLoss={train_loss:.4f} | VaLoss={val_loss:.4f} | "
              f"MeanDice={mean_dice:.4f} | " +
              " ".join(f"{n}:{v:.3f}" for n, v in zip(LABEL_NAMES, dpc)))

    if mean_dice > best_dice:
        best_dice, best_epoch, no_improve = mean_dice, epoch, 0
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "cfg":       json.dumps(CFG),
        }, os.path.join(CFG["output_dir"], SAVE_AS))
        print(f"  ✅ New best Dice={best_dice:.4f} saved to {SAVE_AS}")
    else:
        no_improve += 1

    if epoch % 50 == 0:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg":   json.dumps(CFG),
        }, os.path.join(CFG["output_dir"], f"checkpoint_v2_ep{epoch}.pth"))

    if no_improve >= CFG["patience"]:
        print(f"\n⏹️  Early stop at epoch {epoch} "
              f"(no improvement for {CFG['patience']} epochs)")
        break

print(f"\n✅ Best Mean Dice: {best_dice:.4f} at epoch {best_epoch}")

# =====================================================================
# STEP 8 — PLOTS
# =====================================================================
with open(os.path.join(CFG["output_dir"], "history_v2.json"), "w") as f:
    json.dump(history, f, indent=2)

ep = list(range(start_epoch, start_epoch + len(history["train_loss"])))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(ep, history["train_loss"], label="Train")
axes[0].plot(ep, history["val_loss"],   label="Val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(ep, history["val_dice_mean"], color="green")
axes[1].axhline(best_dice, color="red", linestyle="--",
                label=f"Best={best_dice:.4f} @ep{best_epoch}")
axes[1].set_title("Val Mean Dice"); axes[1].legend(); axes[1].set_xlabel("Epoch")

colors = ["red", "lime", "blue", "yellow", "magenta"]
for n, col in zip(LABEL_NAMES, colors):
    axes[2].plot(ep, history[f"val_dice_{n}"], label=n, color=col)
axes[2].set_title("Per-class Dice"); axes[2].legend(fontsize=8)
axes[2].set_xlabel("Epoch")

plt.suptitle(f"Pituitary 2D UNet V2 — Best={best_dice:.4f} @ep{best_epoch}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CFG["output_dir"], "training_curves_v2.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# =====================================================================
# STEP 9 — FINAL PER-PATIENT EVALUATION
# =====================================================================
print("\n" + "="*75)
print("FINAL PER-PATIENT EVALUATION (best_model_v2)")
print("="*75)

ckpt_v2 = torch.load(os.path.join(CFG["output_dir"], SAVE_AS),
                     map_location=device, weights_only=False)
model.load_state_dict(ckpt_v2["model"])
model.eval()

from collections import defaultdict
patient_slices = defaultdict(list)
for i, s in enumerate(val_slices):
    patient_slices[s["pid"]].append(i)

print(f"\n{'PID':<12} {'Tumor':>7} {'Gland':>7} {'ICA-L':>7} "
      f"{'ICA-R':>7} {'Chiasm':>7} {'Mean':>7}")
print("-"*58)

all_dices = []
with torch.no_grad():
    for pid, indices in sorted(patient_slices.items()):
        imgs = torch.stack(
            [torch.from_numpy(val_slices[i]["image"]).float() for i in indices]
        ).to(device)
        lbls = torch.stack(
            [torch.from_numpy(val_slices[i]["label"]).long() for i in indices]
        ).to(device)

        preds = torch.cat([
            torch.argmax(model(imgs[s:s+CFG["batch_size"]]), dim=1)
            for s in range(0, len(imgs), CFG["batch_size"])
        ], dim=0)

        row = []
        for cls in range(1, CFG["num_classes"]):
            p = (preds == cls).float()
            g = (lbls  == cls).float()
            denom = p.sum() + g.sum()
            row.append((2*(p*g).sum()/denom).item() if denom > 0 else float("nan"))

        all_dices.append(row)
        vals = "  ".join(f"{v:7.3f}" if not np.isnan(v) else "    nan" for v in row)
        print(f"{pid:<12} {vals}  {np.nanmean(row):7.3f}")

arr = np.array(all_dices)
print("-"*58)
print("Mean        " + "  ".join(f"{m:7.3f}" for m in np.nanmean(arr, axis=0)) +
      f"  {np.nanmean(arr):7.3f}")
print("Std         " + "  ".join(f"{s:7.3f}" for s in np.nanstd(arr,  axis=0)))
print(f"\n✅ All done! V2 model saved to: {os.path.join(CFG['output_dir'], SAVE_AS)}")
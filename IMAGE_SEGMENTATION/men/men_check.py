import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, EnsureTyped,
    CastToTyped, Lambdad, AsDiscrete,
    RandGaussianNoised, RandAdjustContrastd,
    RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SegResNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# ================= CONFIG =================

dataset_root  = "/home/23bq1a4293/Personal/project/seg_data/meningioma/BraTS-MEN-Train"
original_file = "top_400_patients.txt"
sampled_file  = "sampled_300_patients.txt"
checkpoint    = "best_model_raw_4class.pth"   # resume from best raw checkpoint

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size  = 1
patch_size  = (192, 192, 128)
num_epochs  = 250
start_epoch = 150       # resume from epoch 96
lr          = 1e-5     # lower LR for fine-tuning from checkpoint

NUM_CLASSES = 4

# ================= COMBINE 400 + 300 = 700 =================

with open(original_file) as f:
    original_ids = [l.strip() for l in f if l.strip()]
with open(sampled_file) as f:
    sampled_ids = [l.strip() for l in f if l.strip()]

patient_ids = sorted(set(original_ids + sampled_ids))
print(f"Original : {len(original_ids)}")
print(f"Sampled  : {len(sampled_ids)}")
print(f"Combined : {len(patient_ids)}")

# ================= LOAD DATA =================

data_dicts = []
for pid in patient_ids:
    case_path   = os.path.join(dataset_root, pid)
    image_paths = [
        os.path.join(case_path, f"{pid}-t1c.nii.gz"),
        os.path.join(case_path, f"{pid}-t1n.nii.gz"),
        os.path.join(case_path, f"{pid}-t2f.nii.gz"),
        os.path.join(case_path, f"{pid}-t2w.nii.gz"),
    ]
    label_path = os.path.join(case_path, f"{pid}-seg.nii.gz")
    if all(os.path.exists(p) for p in image_paths) and os.path.exists(label_path):
        data_dicts.append({"image": image_paths, "label": label_path})

print(f"Total valid cases : {len(data_dicts)}")
train_files, val_files = train_test_split(data_dicts, test_size=0.1, random_state=42)
print(f"Train : {len(train_files)} | Val : {len(val_files)}")

# ================= TRANSFORMS =================

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    CastToTyped(keys=["label"], dtype=torch.int64),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                         b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=patch_size, pos=2, neg=1, num_samples=2,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5)),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    CastToTyped(keys=["label"], dtype=torch.int64),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                         b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = Dataset(train_files, train_transforms)
val_ds   = Dataset(val_files,   val_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=4)

# ================= MODEL =================

model = SegResNet(
    spatial_dims=3, in_channels=4,
    out_channels=NUM_CLASSES, init_filters=32
).to(device)

# resume from checkpoint
assert os.path.exists(checkpoint), f"Checkpoint not found: {checkpoint}"
model.load_state_dict(torch.load(checkpoint, map_location=device))
print(f"Resumed from: {checkpoint} at epoch {start_epoch}")

loss_function = DiceFocalLoss(
    to_onehot_y=True, softmax=True, include_background=False,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

# cosine annealing for remaining epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=(num_epochs - start_epoch),   # remaining epochs only
    eta_min=1e-7,
)

dice_metric     = DiceMetric(include_background=False, reduction="mean")
dice_metric_all = DiceMetric(include_background=False, reduction="none")
scaler          = torch.amp.GradScaler("cuda")
post_pred       = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label      = AsDiscrete(to_onehot=NUM_CLASSES)

# ================= SANITY CHECK =================

print("\n--- Sanity check ---")
_s = val_ds[0]
print(f"  Image : {_s['image'].shape} | min={_s['image'].min():.3f} max={_s['image'].max():.3f}")
print(f"  Label : {_s['label'].shape} | unique={_s['label'].unique().tolist()}")
print("---\n")

# ================= TRAIN =================

best_smoothed_dice = 0.708   # start from known best so we don't overwrite with worse
best_raw_dice      = 0.0
val_dice_history   = []

for epoch in range(start_epoch, num_epochs):

    # ===== TRAINING =====
    model.train()
    epoch_loss = 0.0

    for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1} train", leave=False):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss    = loss_function(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # ===== VALIDATION — overlap=0.25 for speed =====
    model.eval()
    dice_metric.reset()
    dice_metric_all.reset()
    per_case_dices = []

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            val_outputs = sliding_window_inference(
                val_inputs, roi_size=patch_size, sw_batch_size=1,
                predictor=model, overlap=0.25   # faster validation
            )

            val_outputs_list = [post_pred(i)  for i in decollate_batch(val_outputs)]
            val_labels_list  = [post_label(i) for i in decollate_batch(val_labels)]

            dice_metric(val_outputs_list, val_labels_list)
            dice_metric_all(val_outputs_list, val_labels_list)

            case_vals = [x for x in dice_metric_all.aggregate().flatten().tolist() if x == x]
            per_case_dices.append(sum(case_vals) / len(case_vals) if case_vals else 0.0)
            dice_metric_all.reset()

    metric = dice_metric.aggregate().item()
    dice_metric.reset()

    per_case_dices = sorted(per_case_dices)
    worst_5 = sum(per_case_dices[:5])  / 5
    median  = per_case_dices[len(per_case_dices) // 2]
    best_5  = sum(per_case_dices[-5:]) / 5

    val_dice_history.append(metric)
    smoothed_dice = sum(val_dice_history[-3:]) / min(len(val_dice_history), 3)

    if smoothed_dice > best_smoothed_dice:
        best_smoothed_dice = smoothed_dice
        torch.save(model.state_dict(), "best_model_smoothed_4class.pth")
        print(f"  --> Best smoothed saved ({smoothed_dice:.6f})")

    if metric > best_raw_dice:
        best_raw_dice = metric
        torch.save(model.state_dict(), "best_model_raw_4class.pth")
        print(f"  --> Best raw saved ({metric:.6f})")

    print(
        f"Epoch {epoch+1:>3}/{num_epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Dice: {metric:.4f} | "
        f"Smoothed: {smoothed_dice:.4f} | "
        f"Worst5: {worst_5:.4f} | "
        f"Median: {median:.4f} | "
        f"Best5: {best_5:.4f} | "
        f"LR: {current_lr:.2e}"
    )
import os
import math
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils import MetricReduction
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

from IMAGE_SEGMENTATION.glioma.python_scripts.dataset import build_dataset_dicts, get_train_val_transforms, MultiTaskBraTSDataset

# ───────────────────────────── CONFIG ─────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS        = 150          # SegResNet is highly efficient, 150 epochs is plenty
LR            = 1e-4
MIN_LR        = 1e-6
WARMUP_EPOCHS = 10

START_EPOCH   = 0
RESUME_FROM   = None

SAVE_BEST_MEAN = "segresnet_best_mean.pth"
CKPT_DIR       = "checkpoints_segresnet"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_EVERY     = 10

FOLDER_MAP = {
    "/home/23bq1a4293/Personal/project/seg_data/glioma/MICCAI_BraTS_2019_Data_Training/HGG": 0,
    "/home/23bq1a4293/Personal/project/seg_data/glioma/MICCAI_BraTS_2019_Data_Training/LGG": 1,
}
# ──────────────────────────────────────────────────────────────────────────────

# Switching to DiceFocalLoss to heavily penalize missing small Tumor pixels (ET)
dice_focal = DiceFocalLoss(
    include_background=True, sigmoid=True,
    squared_pred=True, reduction="mean", gamma=2.0
)

def get_lr(epoch, total_epochs, base_lr, min_lr, warmup_epochs):
    if epoch < warmup_epochs:
        return base_lr * ((epoch + 1) / warmup_epochs)
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    cosine   = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(cosine, min_lr)

def eval_model(model, val_loader, dice_metric):
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for images, seg_labels, _ in val_loader:
            images, seg_labels = images.to(DEVICE), seg_labels.to(DEVICE)

            with torch.amp.autocast("cuda"):
                seg_final = sliding_window_inference(
                    inputs=images,
                    roi_size=(128, 128, 128),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                )
            preds_bin = (torch.sigmoid(seg_final) > 0.5).float()
            dice_metric(y_pred=preds_bin, y=seg_labels)

    dice_scores = dice_metric.aggregate()
    tc, wt, et = dice_scores[0].item(), dice_scores[1].item(), dice_scores[2].item()
    mn = dice_scores.mean().item()
    dice_metric.reset()
    return tc, wt, et, mn

def train_segresnet():
    print(f"Using device: {DEVICE}")

    data_dicts = build_dataset_dicts(FOLDER_MAP)
    train_tf, val_tf = get_train_val_transforms()
    split = int(len(data_dicts) * 0.8)
    
    train_ds = MultiTaskBraTSDataset(data_dicts[:split], transform=train_tf)
    val_ds   = MultiTaskBraTSDataset(data_dicts[split:], transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(data_dicts[:split])} | Val: {len(data_dicts[split:])} patients")

    # SegResNet Initialization (The CNN Master of BraTS)
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(DEVICE)

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        model.load_state_dict(torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True))
        print(f"Loaded weights from {RESUME_FROM}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)

    best_mean = 0.0

    print("\n[Starting SegResNet Training...]")
    for epoch in range(START_EPOCH, EPOCHS):
        lr_now = get_lr(epoch, EPOCHS, LR, MIN_LR, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()
        epoch_loss = 0.0

        for step, (images, seg_labels, _) in enumerate(train_loader, 1):
            images, seg_labels = images.to(DEVICE), seg_labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                seg_final = model(images)
                loss = dice_focal(seg_final, seg_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if step % 50 == 0 or step == len(train_loader):
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {lr_now:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1} Avg Train Loss: {avg_loss:.4f}")

        tc, wt, et, mn = eval_model(model, val_loader, dice_metric)
        print(f"Val Dice  =>  TC: {tc:.4f} | WT: {wt:.4f} | ET: {et:.4f} | Mean: {mn:.4f}")
        print(f"{'─'*60}\n")

        if mn > best_mean:
            best_mean = mn
            torch.save(model.state_dict(), SAVE_BEST_MEAN)
            print(f"=> [BEST MEAN] Saved  (Mean: {mn:.4f} | TC: {tc:.4f} | WT: {wt:.4f} | ET: {et:.4f})\n")

        if (epoch + 1) % CKPT_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"epoch_{epoch+1:03d}.pth"))

    print(f"\nTraining complete! Best Mean Dice: {best_mean:.4f}")

if __name__ == "__main__":
    train_segresnet()
import glob
import os
import random
import time
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet
import numpy as np
import scipy.ndimage as ndi
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --- Configuration ---
DATA_DIR = (
    "/home/23bq1a4293/Personal/project/seg_data/pit_adv/processed_sella_05mm"
)
OUTPUT_DIR = "/home/23bq1a4293/Personal/project/seg_data/pit_adv/pituitary_sella_zoom_output"
PRETRAINED_WEIGHTS = "/home/23bq1a4293/Personal/project/seg_data/pit_adv/pituitary_dual_output/best_pituitary_dual_sota.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
LR = 1.5e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2
INPUT_SIZE = (128, 128, 128)  # 64mm FOV at 0.5mm
NUM_CLASSES = 3  # 0: BG, 1: Tumor, 2: Carotids

os.makedirs(OUTPUT_DIR, exist_ok=True)


class SellaZoomDataset(Dataset):

  def __init__(self, file_paths, is_train=True):
    self.file_paths = file_paths
    self.is_train = is_train

  def __len__(self):
    return len(self.file_paths)

  def __getitem__(self, idx):
    data = torch.load(self.file_paths[idx])
    img = data["image"].float()  # [2, 160, 160, 160]
    lbl = data["label"].long()  # [1, 160, 160, 160]

    _, D, H, W = img.shape
    tD, tH, tW = INPUT_SIZE

    if self.is_train:
      # Random spatial jitter crop (+/- 16 voxels around center)
      sd = random.randint(0, D - tD)
      sh = random.randint(0, H - tH)
      sw = random.randint(0, W - tW)
    else:
      # Deterministic center crop for validation
      sd = (D - tD) // 2
      sh = (H - tH) // 2
      sw = (W - tW) // 2

    img = img[:, sd : sd + tD, sh : sh + tH, sw : sw + tW]
    lbl = lbl[:, sd : sd + tD, sh : sh + tH, sw : sw + tW]

    # Data augmentations
    if self.is_train:
      if random.random() > 0.5:
        img = torch.flip(img, dims=[1])
        lbl = torch.flip(lbl, dims=[1])
      if random.random() > 0.5:
        img = torch.flip(img, dims=[2])
        lbl = torch.flip(lbl, dims=[2])
      if random.random() > 0.5:
        img = torch.flip(img, dims=[3])
        lbl = torch.flip(lbl, dims=[3])
      if random.random() > 0.3:
        img = img * random.uniform(0.9, 1.1)

    return img, lbl, data["case_id"]


def compute_dice(pred_binary, gt_binary):
  intersection = np.logical_and(pred_binary, gt_binary).sum()
  total = pred_binary.sum() + gt_binary.sum()
  if total == 0:
    return 1.0
  return float(2.0 * intersection / (total + 1e-6))


def clean_carotid_prediction(pred_mask, min_voxels=50):
  carotid_mask = pred_mask == 2
  if not np.any(carotid_mask):
    return pred_mask
  labeled, num_features = ndi.label(carotid_mask)
  if num_features == 0:
    return pred_mask
  component_sizes = np.bincount(labeled.ravel())
  clean_mask = np.copy(pred_mask)
  for comp_idx in range(1, num_features + 1):
    if component_sizes[comp_idx] < min_voxels:
      clean_mask[labeled == comp_idx] = 0
  return clean_mask


def main():
  torch.backends.cudnn.benchmark = True
  print(f"[{time.strftime('%H:%M:%S')}] Active Device: {DEVICE}")

  all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
  if len(all_files) == 0:
    raise RuntimeError(f"No preprocessed files found in {DATA_DIR}!")

  random.seed(42)
  shuffled = list(all_files)
  random.shuffle(shuffled)
  train_files = shuffled[:80]
  val_files = shuffled[80:]

  print(
      f"Total Cases: {len(all_files)} | Train: {len(train_files)} | Val:"
      f" {len(val_files)}"
  )

  train_loader = DataLoader(
      SellaZoomDataset(train_files, is_train=True),
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=4,
      pin_memory=True,
  )
  val_loader = DataLoader(
      SellaZoomDataset(val_files, is_train=False),
      batch_size=1,
      shuffle=False,
      num_workers=2,
  )

  # 18.8M SegResNet
  model = SegResNet(
      spatial_dims=3,
      in_channels=2,
      out_channels=NUM_CLASSES,
      init_filters=32,
      blocks_down=(1, 2, 2, 4),
      blocks_up=(1, 1, 1),
      dropout_prob=0.1,
  ).to(DEVICE)

  # Load Pretrained 1.0mm Dual Weights if available
  if os.path.exists(PRETRAINED_WEIGHTS):
    print(f"Loading pretrained weights from {PRETRAINED_WEIGHTS}...")
    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
    state_dict = (
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    )
    model.load_state_dict(state_dict)
    print("Pretrained Dual-Modality weights transferred successfully!")

  loss_fn = DiceCELoss(
      to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=EPOCHS, eta_min=1e-6
  )
  scaler = GradScaler("cuda")

  best_mean_dice = 0.0

  print("=" * 80)
  print(
      f"LAUNCHING NATIVE 0.5mm SELLA-ZOOM SOTA TRAINING (Epochs: {EPOCHS} |"
      f" Input: {INPUT_SIZE})"
  )
  print("=" * 80)

  for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    step_count = 0
    t_start = time.time()
    optimizer.zero_grad()

    for step, (images, labels, _) in enumerate(train_loader):
      images = images.to(DEVICE, non_blocking=True)
      labels = labels.to(DEVICE, non_blocking=True)

      with autocast("cuda"):
        logits = model(images)
        loss = loss_fn(logits, labels) / GRAD_ACCUM_STEPS

      scaler.scale(loss).backward()
      epoch_loss += loss.item() * GRAD_ACCUM_STEPS
      step_count += 1

      if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    scheduler.step()
    avg_train_loss = epoch_loss / max(1, step_count)
    epoch_duration = time.time() - t_start

    # Validate every 2 epochs
    if epoch % 2 == 0 or epoch == EPOCHS:
      model.eval()
      tumor_dices, carotid_dices, clean_carotid_dices = [], [], []

      with torch.no_grad():
        for images, labels, _ in val_loader:
          images = images.to(DEVICE)
          gt = labels.squeeze().cpu().numpy()

          with autocast("cuda"):
            logits = model(images)
            preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

          t_d = compute_dice(preds == 1, gt == 1)
          c_d = compute_dice(preds == 2, gt == 2)
          cleaned = clean_carotid_prediction(preds, min_voxels=50)
          cl_c_d = compute_dice(cleaned == 2, gt == 2)

          tumor_dices.append(t_d)
          carotid_dices.append(c_d)
          clean_carotid_dices.append(cl_c_d)

      mean_tumor = np.mean(tumor_dices)
      mean_carotid = np.mean(carotid_dices)
      mean_clean_carotid = np.mean(clean_carotid_dices)
      mean_dice = (mean_tumor + mean_clean_carotid) / 2.0

      log_str = (
          f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:02d}/{EPOCHS}"
          f" ({epoch_duration:.1f}s) | Loss: {avg_train_loss:.4f} | Tumor:"
          f" {mean_tumor:.4f} | Carotids: {mean_carotid:.4f} (Clean:"
          f" {mean_clean_carotid:.4f}) | Mean: {mean_dice:.4f}"
      )

      if mean_dice > best_mean_dice:
        best_mean_dice = mean_dice
        ckpt_path = os.path.join(
            OUTPUT_DIR, "best_pituitary_sella_zoom_05mm.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_mean_dice": best_mean_dice,
                "tumor_dice": mean_tumor,
                "carotid_dice": mean_clean_carotid,
            },
            ckpt_path,
        )
        log_str += f" -> [NEW SOTA SAVED: {best_mean_dice:.4f}]"

      print(log_str)
    else:
      print(
          f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:02d}/{EPOCHS}"
          f" ({epoch_duration:.1f}s) | Train Loss: {avg_train_loss:.4f}"
      )

  print("=" * 80)
  print(
      "[DONE] 0.5mm Sella-Centric Training Complete! Peak SOTA Mean Dice:"
      f" {best_mean_dice:.4f}"
  )
  print(
      "Checkpoint saved at:"
      f" {os.path.join(OUTPUT_DIR, 'best_pituitary_sella_zoom_05mm.pth')}"
  )


if __name__ == "__main__":
  main()
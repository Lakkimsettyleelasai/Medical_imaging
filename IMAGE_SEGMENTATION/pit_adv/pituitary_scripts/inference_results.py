import glob
import os
import random
import time
from monai.networks.nets import SegResNet
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from torch.amp import autocast

# Headless matplotlib backend for high-speed rendering
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Paths & Config ---
DATA_DIR = (
    '/home/23bq1a4293/Personal/project/seg_data/pit_adv/processed_sella_05mm'
)
CKPT_PATH = '/home/23bq1a4293/Personal/project/seg_data/pit_adv/pituitary_sella_zoom_output/best_pituitary_sella_zoom_05mm.pth'
RESULTS_DIR = '/home/23bq1a4293/Personal/project/IMAGE_SEGMENTATION/pit_adv/results'
CSV_PATH = os.path.join(RESULTS_DIR, 'all_patients_benchmark_summary.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_dice(pred, gt):
  inter = np.logical_and(pred, gt).sum()
  tot = pred.sum() + gt.sum()
  if tot == 0:
    return 1.0
  return float(2.0 * inter / (tot + 1e-6))


def clean_carotids(pred_mask, min_voxels=50):
  """Removes isolated noisy speckles < 50 voxels."""
  carotids = pred_mask == 2
  if not np.any(carotids):
    return pred_mask
  labeled, num = ndi.label(carotids)
  sizes = np.bincount(labeled.ravel())
  cleaned = np.copy(pred_mask)
  for i in range(1, num + 1):
    if sizes[i] < min_voxels:
      cleaned[labeled == i] = 0
  return cleaned


def save_patient_visualization(
    case_id, img_v, gt_v, pred_v, t_dice, c_dice, m_dice, split_name, out_path
):
  t1_v = img_v[0]  # T1-CE
  t2_v = img_v[1]  # T2

  # Find center of the tumor or carotid complex
  fg_coords = np.where(gt_v > 0)
  if len(fg_coords[0]) > 0:
    cz = int(np.mean(fg_coords[0]))
    cy = int(np.mean(fg_coords[1]))
    cx = int(np.mean(fg_coords[2]))
  else:
    cz, cy, cx = 64, 64, 64

  fig, axes = plt.subplots(3, 4, figsize=(18, 13))
  views = [
      ('Axial (Z)', t1_v[cz], t2_v[cz], gt_v[cz], pred_v[cz]),
      ('Coronal (Y)', t1_v[:, cy, :], t2_v[:, cy, :], gt_v[:, cy, :], pred_v[:, cy, :]),
      ('Sagittal (X)', t1_v[:, :, cx], t2_v[:, :, cx], gt_v[:, :, cx], pred_v[:, :, cx]),
  ]

  for r, (view_name, t1_sl, t2_sl, gt_sl, pred_sl) in enumerate(views):
    # 1. T1-CE
    axes[r, 0].imshow(t1_sl, cmap='gray', origin='lower')
    axes[r, 0].set_title(f'{view_name} - T1-CE Native 0.5mm', fontsize=10)
    axes[r, 0].axis('off')

    # 2. T2
    axes[r, 1].imshow(t2_sl, cmap='gray', origin='lower')
    axes[r, 1].set_title(f'{view_name} - T2 Co-Registered', fontsize=10)
    axes[r, 1].axis('off')

    # 3. Ground Truth Overlay (Red: Tumor, Cyan: Carotids)
    axes[r, 2].imshow(t1_sl, cmap='gray', origin='lower')
    gt_rgb = np.zeros((*gt_sl.shape, 4))
    gt_rgb[gt_sl == 1] = [1.0, 0.1, 0.1, 0.55]  # Red
    gt_rgb[gt_sl == 2] = [0.0, 0.9, 1.0, 0.70]  # Cyan
    axes[r, 2].imshow(gt_rgb, origin='lower')
    axes[r, 2].set_title(f'{view_name} - Ground Truth', fontsize=10)
    axes[r, 2].axis('off')

    # 4. SOTA AI Prediction Overlay (Green: Tumor, Cyan: Carotids)
    axes[r, 3].imshow(t1_sl, cmap='gray', origin='lower')
    pred_rgb = np.zeros((*pred_sl.shape, 4))
    pred_rgb[pred_sl == 1] = [0.0, 1.0, 0.2, 0.55]  # Green
    pred_rgb[pred_sl == 2] = [0.0, 0.9, 1.0, 0.70]  # Cyan
    axes[r, 3].imshow(pred_rgb, origin='lower')
    axes[r, 3].set_title(f'{view_name} - 0.5mm SOTA Prediction', fontsize=10)
    axes[r, 3].axis('off')

  plt.suptitle(
      f'Patient {case_id} [{split_name}] | Tumor Dice: {t_dice:.4f} | Carotid'
      f' Dice: {c_dice:.4f} | Mean: {m_dice:.4f}',
      fontsize=14,
      fontweight='bold',
  )
  plt.tight_layout()
  plt.savefig(out_path, dpi=200, bbox_inches='tight')
  plt.close(fig)


def main():
  print('=' * 85)
  print('LAUNCHING FULL COHORT EVALUATION & VISUALIZATION (All Patients)')
  print(f'Device: {DEVICE} | Output Directory: {RESULTS_DIR}')
  print('=' * 85)

  # Load model
  print(f'Loading weights from: {CKPT_PATH} ...')
  ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
  state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

  model = SegResNet(
      spatial_dims=3,
      in_channels=2,
      out_channels=3,
      init_filters=32,
      blocks_down=(1, 2, 2, 4),
      blocks_up=(1, 1, 1),
  ).to(DEVICE)
  model.load_state_dict(state_dict)
  model.eval()

  # Gather all cases
  all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.pt')))
  print(f'Found {len(all_files)} total preprocessed cases.')

  # Identify validation vs train split (same seed=42)
  random.seed(42)
  shuffled = list(all_files)
  random.shuffle(shuffled)
  val_set = set(shuffled[80:])

  records = []
  t0 = time.time()

  with torch.no_grad():
    for idx, f in enumerate(all_files):
      data = torch.load(f)
      case_id = data['case_id']
      split_name = 'Validation (Unseen)' if f in val_set else 'Train'

      img = data['image'].float()  # [2, 160, 160, 160]
      gt = data['label'].long().squeeze().numpy()

      # Center crop 128^3
      _, D, H, W = img.shape
      sd, sh, sw = (D - 128) // 2, (H - 128) // 2, (W - 128) // 2
      img_crop = (
          img[:, sd : sd + 128, sh : sh + 128, sw : sw + 128]
          .unsqueeze(0)
          .to(DEVICE)
      )
      gt_crop = gt[sd : sd + 128, sh : sh + 128, sw : sw + 128]

      with autocast('cuda'):
        logits = model(img_crop)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

      cleaned_pred = clean_carotids(pred, min_voxels=50)

      # Dices
      t_dice = compute_dice(cleaned_pred == 1, gt_crop == 1)
      c_dice = compute_dice(cleaned_pred == 2, gt_crop == 2)
      m_dice = (t_dice + c_dice) / 2.0

      # Volumes in cm3 (each voxel at 0.5mm is 0.5*0.5*0.5 = 0.125 mm3 = 0.000125 cm3)
      tumor_vol_cm3 = (cleaned_pred == 1).sum() * 0.000125
      carotid_vol_cm3 = (cleaned_pred == 2).sum() * 0.000125

      records.append({
          'Case_ID': case_id,
          'Split': split_name,
          'Tumor_Dice': round(t_dice, 4),
          'Carotid_Dice': round(c_dice, 4),
          'Mean_Dice': round(m_dice, 4),
          'Tumor_Vol_cm3': round(tumor_vol_cm3, 3),
          'Carotid_Vol_cm3': round(carotid_vol_cm3, 3),
      })

      # Save individual multi-planar slice PNG
      out_png = os.path.join(RESULTS_DIR, f'case_{case_id}_vis.png')
      save_patient_visualization(
          case_id,
          img_crop[0].cpu().numpy(),
          gt_crop,
          cleaned_pred,
          t_dice,
          c_dice,
          m_dice,
          split_name,
          out_png,
      )

      if (idx + 1) % 10 == 0 or idx == len(all_files) - 1:
        elapsed = time.time() - t0
        print(
            f'[{idx+1:02d}/{len(all_files)}] Processed & Visualized {case_id} |'
            f' Tumor: {t_dice:.3f} | Carotid: {c_dice:.3f} | Elapsed:'
            f' {elapsed:.1f}s'
        )

  # Build DataFrame
  df = pd.DataFrame(records)
  df.to_csv(CSV_PATH, index=False)
  print(f'\n[SUCCESS] Summary spreadsheet saved to: {CSV_PATH}')

  # Split summaries
  val_df = df[df['Split'] == 'Validation (Unseen)']
  train_df = df[df['Split'] == 'Train']

  print('\n' + '=' * 85)
  print('FINAL COHORT BENCHMARK RESULTS')
  print('=' * 85)

  print(f'1. UNSEEN TEST/VALIDATION COHORT ({len(val_df)} Patients):')
  print(
      f"   - Total Mean Tumor Dice:       {val_df['Tumor_Dice'].mean():.4f} "
      f" (Peak: {val_df['Tumor_Dice'].max():.4f})"
  )
  print(
      f"   - Total Mean Carotid Dice:     {val_df['Carotid_Dice'].mean():.4f} "
      f" (Peak: {val_df['Carotid_Dice'].max():.4f})"
  )
  print(
      f"   - Total Overall Mean Dice:     {val_df['Mean_Dice'].mean():.4f} "
      f" (Peak: {val_df['Mean_Dice'].max():.4f})"
  )

  print(f'\n2. TRAINING COHORT ({len(train_df)} Patients):')
  print(
      f"   - Total Mean Tumor Dice:       {train_df['Tumor_Dice'].mean():.4f}"
  )
  print(
      f"   - Total Mean Carotid Dice:     {train_df['Carotid_Dice'].mean():.4f}"
  )
  print(f"   - Total Overall Mean Dice:     {train_df['Mean_Dice'].mean():.4f}")

  print(f'\n3. TOTAL GRAND COHORT ({len(df)} Patients):')
  print(f"   >>> TOTAL GRAND MEAN TUMOR DICE:   {df['Tumor_Dice'].mean():.4f}")
  print(
      f"   >>> TOTAL GRAND MEAN CAROTID DICE: {df['Carotid_Dice'].mean():.4f}"
  )
  print(f"   >>> TOTAL GRAND OVERALL MEAN DICE: {df['Mean_Dice'].mean():.4f}")
  print('=' * 85)
  print(
      f'All {len(df)} visualization files are ready in: file://{RESULTS_DIR}/\n'
  )


if __name__ == '__main__':
  main()
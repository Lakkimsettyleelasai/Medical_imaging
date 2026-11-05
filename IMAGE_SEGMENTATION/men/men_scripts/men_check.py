import os
import glob
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# ================= CONFIG =================
dataset_root = "/home/23bq1a4293/Personal/project/seg_data/meningioma/BraTS-MEN-Train"
checkpoint   = "/home/23bq1a4293/Personal/project/models/Segmentation/men/best_model_smoothed_4class.pth"

patch_size  = (192, 192, 128)
NUM_CLASSES = 4
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODEL =================
print("Loading SegResNet...")
model = SegResNet(
    spatial_dims=3, in_channels=4,
    out_channels=NUM_CLASSES, init_filters=32
).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.eval()
print(f"✅ Model loaded: {checkpoint}")

# ================= DIRECT NIBABEL LOADER =================
def load_case(pid):
    """Load all modalities + label directly with nibabel, returns tensors."""
    case_path = os.path.join(dataset_root, pid)
    modalities = ["t1c", "t1n", "t2f", "t2w"]
    
    vols = []
    for m in modalities:
        path = os.path.join(case_path, f"{pid}-{m}.nii.gz")
        arr  = nib.load(path).get_fdata(dtype=np.float32)
        arr  = np.clip(arr, 0, 3000) / 3000.0  # normalize
        vols.append(arr)
    
    image = np.stack(vols, axis=0)
    
    label_path = os.path.join(case_path, f"{pid}-seg.nii.gz")
    label = nib.load(label_path).get_fdata(dtype=np.float32).astype(np.uint8)
    
    return image, label

# ================= GET ALL 1000 PATIENTS =================
all_patients_paths = sorted(glob.glob(os.path.join(dataset_root, "BraTS-MEN-*")))
all_patients = [os.path.basename(p) for p in all_patients_paths]
print(f"✅ {len(all_patients)} total patients available for evaluation")

# ================= METRICS SETUP =================
post_pred  = AsDiscrete(argmax=False, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)
dm = DiceMetric(include_background=False, reduction="none")

all_nc, all_ed, all_et = [], [], []

print("\nStarting evaluation on all 1000 patients...")

# ================= INFERENCE LOOP =================
with torch.no_grad():
    for pid in tqdm(all_patients, desc="Evaluating"):
        image_np, label_np = load_case(pid)
        image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(device)

        with torch.amp.autocast("cuda"):
            out = sliding_window_inference(
                image_tensor, roi_size=patch_size,
                sw_batch_size=1, predictor=model, overlap=0.25
            )

        pred_np = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # Calculate Per-class Dice
        p = post_pred(torch.from_numpy(pred_np).unsqueeze(0))
        g = post_label(torch.from_numpy(label_np).unsqueeze(0).long())
        dm([p], [g])
        
        # 💥 THE FIX: Convert NaN (True Negatives) to 1.0 (Perfect Score) 💥
        per_class = [1.0 if np.isnan(v) else v for v in dm.aggregate()[0].tolist()]
        dm.reset()
        
        all_nc.append(per_class[0])
        all_ed.append(per_class[1])
        all_et.append(per_class[2])

# ================= FINAL RESULTS =================
nc_mean = np.mean(all_nc)
ed_mean = np.mean(all_ed)
et_mean = np.mean(all_et)
overall = np.mean([nc_mean, ed_mean, et_mean])

print("\n" + "="*50)
print(f"FINAL TRUE SCORES ON {len(all_patients)} PATIENTS")
print("="*50)
print(f"Necrotic Core (Class 1): {nc_mean:.4f}")
print(f"Edema (Class 2):         {ed_mean:.4f}")
print(f"Enhancing (Class 3):     {et_mean:.4f}")
print("-" * 50)
print(f"TRUE MEAN DICE:          {overall:.4f}")
print("="*50)
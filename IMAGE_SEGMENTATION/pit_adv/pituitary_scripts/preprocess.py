import os
import glob
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

RAW_DATA_DIR = "/home/23bq1a4293/Personal/project/seg_data/pit_adv"
OUT_DIR = "/home/23bq1a4293/Personal/project/seg_data/pit_adv/processed_sella_05mm"
TARGET_SPACING = (0.5, 0.5, 0.5)
CROP_SIZE = (160, 160, 160)  # 80mm x 80mm x 80mm around sella

os.makedirs(OUT_DIR, exist_ok=True)


def resample_to_spacing(data, original_spacing, target_spacing, is_mask=False):
    """Resample 3D numpy array to target voxel spacing."""
    scale_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    if np.allclose(scale_factors, [1.0, 1.0, 1.0], atol=1e-2):
        return data

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
    mode = "nearest" if is_mask else "trilinear"
    align_corners = None if is_mask else False

    resampled = F.interpolate(
        tensor,
        scale_factor=scale_factors,
        mode=mode,
        align_corners=align_corners,
    )
    return resampled.squeeze(0).squeeze(0).numpy()


def normalize_intensity(volume):
    """Non-zero z-score standardization."""
    mask = volume > 0
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32)
    mean = np.mean(volume[mask])
    std = np.std(volume[mask])
    if std < 1e-5:
        std = 1.0
    norm = np.zeros_like(volume, dtype=np.float32)
    norm[mask] = (volume[mask] - mean) / std
    return norm


def extract_sella_cube(
    t1_data, t2_data, label_data, crop_shape=(160, 160, 160)
):
    """Finds the center of the sella region (tumor + carotids) and extracts a fixed-size cube with padding if necessary."""
    fg_indices = np.where(label_data > 0)
    if len(fg_indices[0]) > 0:
        zc = int(np.mean(fg_indices[0]))
        yc = int(np.mean(fg_indices[1]))
        xc = int(np.mean(fg_indices[2]))
    else:
        # Fallback to volume center
        zc, yc, xc = [s // 2 for s in label_data.shape]

    cz, cy, cx = crop_shape
    hz, hy, hx = cz // 2, cy // 2, cx // 2

    z_start, z_end = zc - hz, zc + hz
    y_start, y_end = yc - hy, yc + hy
    x_start, x_end = xc - hx, xc + hx

    # Pad if boundaries extend outside volume
    pad_z_before = max(0, -z_start)
    pad_z_after = max(0, z_end - t1_data.shape[0])
    pad_y_before = max(0, -y_start)
    pad_y_after = max(0, y_end - t1_data.shape[1])
    pad_x_before = max(0, -x_start)
    pad_x_after = max(0, x_end - t1_data.shape[2])

    if any(
        [
            pad_z_before,
            pad_z_after,
            pad_y_before,
            pad_y_after,
            pad_x_before,
            pad_x_after,
        ]
    ):
        t1_data = np.pad(
            t1_data,
            (
                (pad_z_before, pad_z_after),
                (pad_y_before, pad_y_after),
                (pad_x_before, pad_x_after),
            ),
            mode="constant",
            constant_values=0,
        )
        t2_data = np.pad(
            t2_data,
            (
                (pad_z_before, pad_z_after),
                (pad_y_before, pad_y_after),
                (pad_x_before, pad_x_after),
            ),
            mode="constant",
            constant_values=0,
        )
        label_data = np.pad(
            label_data,
            (
                (pad_z_before, pad_z_after),
                (pad_y_before, pad_y_after),
                (pad_x_before, pad_x_after),
            ),
            mode="constant",
            constant_values=0,
        )
        z_start += pad_z_before
        z_end += pad_z_before
        y_start += pad_y_before
        y_end += pad_y_before
        x_start += pad_x_before
        x_end += pad_x_before

    t1_crop = t1_data[z_start:z_end, y_start:y_end, x_start:x_end]
    t2_crop = t2_data[z_start:z_end, y_start:y_end, x_start:x_end]
    lbl_crop = label_data[z_start:z_end, y_start:y_end, x_start:x_end]

    return t1_crop, t2_crop, lbl_crop, (zc, yc, xc)


def main():
    t1_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*_T1.nii.gz")))
    print(
        f"Found {len(t1_files)} candidate cases. Verifying paired files & native 0.5mm geometry..."
    )

    valid_cases = []
    for t1_p in t1_files:
        case_id = os.path.basename(t1_p).replace("_T1.nii.gz", "")
        t2_p = os.path.join(RAW_DATA_DIR, f"{case_id}_T2_f.nii.gz")
        tumor_p = os.path.join(RAW_DATA_DIR, f"{case_id}_tumor_f.nii.gz")
        carotids_p = os.path.join(RAW_DATA_DIR, f"{case_id}_carotids_f.nii.gz")

        if (
            os.path.exists(t2_p)
            and os.path.exists(tumor_p)
            and os.path.exists(carotids_p)
        ):
            valid_cases.append((case_id, t1_p, t2_p, tumor_p, carotids_p))

    print(
        f"Verified {len(valid_cases)} complete 4-file cases for Sella-Zoom Preprocessing."
    )

    t0 = time.time()
    for idx, (case_id, t1_p, t2_p, tumor_p, carotids_p) in enumerate(
        valid_cases
    ):
        out_file = os.path.join(OUT_DIR, f"{case_id}.pt")
        if os.path.exists(out_file):
            continue

        # Load Nifti
        t1_nii = nib.load(t1_p)
        t2_nii = nib.load(t2_p)
        tumor_nii = nib.load(tumor_p)
        carotids_nii = nib.load(carotids_p)

        orig_spacing = tuple(t1_nii.header.get_zooms()[:3])
        t1_arr = t1_nii.get_fdata().astype(np.float32)
        t2_arr = t2_nii.get_fdata().astype(np.float32)
        tumor_arr = (tumor_nii.get_fdata() > 0).astype(np.uint8)
        carotids_arr = (carotids_nii.get_fdata() > 0).astype(np.uint8)

        # Resample to strict 0.5mm if not already
        t1_res = resample_to_spacing(
            t1_arr, orig_spacing, TARGET_SPACING, is_mask=False
        )
        t2_res = resample_to_spacing(
            t2_arr, orig_spacing, TARGET_SPACING, is_mask=False
        )
        tumor_res = resample_to_spacing(
            tumor_arr, orig_spacing, TARGET_SPACING, is_mask=True
        )
        carotids_res = resample_to_spacing(
            carotids_arr, orig_spacing, TARGET_SPACING, is_mask=True
        )

        # Merge masks: 0=BG, 1=Tumor, 2=Carotids (carotids prioritized at intersection)
        label_res = np.zeros_like(tumor_res, dtype=np.int64)
        label_res[tumor_res > 0] = 1
        label_res[carotids_res > 0] = 2

        # Intensity Normalize
        t1_norm = normalize_intensity(t1_res)
        t2_norm = normalize_intensity(t2_res)

        # Extract 160^3 Sella-Centric Cube
        t1_crop, t2_crop, lbl_crop, center = extract_sella_cube(
            t1_norm, t2_norm, label_res, CROP_SIZE
        )

        # Stack into [2, 160, 160, 160] and [1, 160, 160, 160]
        image_tensor = torch.from_numpy(
            np.stack([t1_crop, t2_crop], axis=0)
        ).half()  # half to save disk
        label_tensor = (
            torch.from_numpy(lbl_crop).unsqueeze(0).to(torch.int8)
        )  # int8 saves disk

        torch.save(
            {
                "case_id": case_id,
                "image": image_tensor,
                "label": label_tensor,
                "center": center,
                "orig_shape": t1_arr.shape,
            },
            out_file,
        )

        if (idx + 1) % 10 == 0 or idx == len(valid_cases) - 1:
            elapsed = time.time() - t0
            print(
                f"[{idx+1:03d}/{len(valid_cases)}] Preprocessed {case_id} -> Crop: {t1_crop.shape} | Elapsed: {elapsed:.1f}s"
            )

    print(
        f"\n[DONE] Sella-Centric 0.5mm preprocessing complete! Saved in: {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
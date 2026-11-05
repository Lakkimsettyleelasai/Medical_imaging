import os
import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandGaussianNoised, ToTensord,
    MapTransform, SpatialPadd, RandGaussianSmoothd,
    RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd,
)
from monai.data import Dataset
# ─── BraTS label → WT / TC / ET multi-label mask ────────────────────────────
class ConvertBraTSLabelsd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label  = d[key]                             # (1, H, W, D)
            result = torch.zeros((3, *label.shape[1:]), dtype=torch.float32)
            result[0] = (label == 1) | (label == 4)    # TC
            result[1] = (label == 1) | (label == 2) | (label == 4)  # WT
            result[2] = (label == 4)                   # ET
            d[key]  = result
        return d
# ─── Transforms ─────────────────────────────────────────────────────────────
def get_train_val_transforms():
    keys_img   = ["t1", "t1ce", "t2", "flair"]
    keys_all   = keys_img + ["label"]
    train_transform = Compose([
        LoadImaged(keys=keys_all),
        EnsureChannelFirstd(keys=keys_all, channel_dim="no_channel"),
        Orientationd(keys=keys_all, axcodes="RAS"),
        ConvertBraTSLabelsd(keys=["label"]),
        CropForegroundd(keys=keys_all, source_key="t1ce"),
        SpatialPadd(keys=keys_all, spatial_size=(128, 128, 128)),
        RandCropByPosNegLabeld(
            keys=keys_all, label_key="label",
            spatial_size=(128, 128, 128),
            pos=1, neg=1, num_samples=2,
            image_key="t1ce", image_threshold=0,
        ),
        NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True),
        # ── Geometric augmentations ──────────────────────────────────
        RandFlipd(keys=keys_all,    prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys_all,    prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys_all,    prob=0.5, spatial_axis=2),
        RandRotate90d(keys=keys_all, prob=0.5, max_k=3),
        # ── Intensity augmentations (image only) ─────────────────────
        RandGaussianNoised(keys=keys_img,     prob=0.15, std=0.05),
        RandGaussianSmoothd(keys=keys_img,    prob=0.10,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                            sigma_z=(0.5, 1.0)),
        RandScaleIntensityd(keys=keys_img,    prob=0.25, factors=0.2),
        RandShiftIntensityd(keys=keys_img,    prob=0.25, offsets=0.1),
        RandAdjustContrastd(keys=keys_img,    prob=0.15, gamma=(0.7, 1.5)),
        ToTensord(keys=keys_all),
    ])
    val_transform = Compose([
        LoadImaged(keys=keys_all),
        EnsureChannelFirstd(keys=keys_all, channel_dim="no_channel"),
        Orientationd(keys=keys_all, axcodes="RAS"),
        ConvertBraTSLabelsd(keys=["label"]),
        CropForegroundd(keys=keys_all, source_key="t1ce"),
        NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True),
        ToTensord(keys=keys_all),
    ])
    return train_transform, val_transform
# ─── Multi-task Dataset ──────────────────────────────────────────────────────
class MultiTaskBraTSDataset(Dataset):
    def __init__(self, data_dicts, transform=None):
        super().__init__(data_dicts, transform)
        self.data_dicts = data_dicts
        self.transform  = transform
    def __getitem__(self, index):
        data = self.data_dicts[index]
        if self.transform is not None:
            data = self.transform(data)
        # RandCropByPosNegLabeld returns a list of patches
        if isinstance(data, list):
            data = data[0]
        img = torch.cat(
            [data["t1"], data["t1ce"], data["t2"], data["flair"]], dim=0
        )  # (4, H, W, D)
        seg = data["label"]                                        # (3, H, W, D)
        cls = torch.tensor(data["class_id"], dtype=torch.long)    # scalar
        return img, seg, cls
# ─── Build file-dictionary list ──────────────────────────────────────────────
def build_dataset_dicts(folder_class_map):
    """
    folder_class_map: { "/path/to/HGG": 0, "/path/to/LGG": 1 }
    """
    data_dicts = []
    for folder_path, class_id in folder_class_map.items():
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found — skipping.")
            continue
        patients = [d for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))]
        for p in patients:
            p_dir = os.path.join(folder_path, p)
            files = glob.glob(os.path.join(p_dir, "*.nii*"))
            def pick(kw, ex=None):
                return next(
                    (f for f in files
                     if kw in os.path.basename(f).lower()
                     and (ex is None or ex not in os.path.basename(f).lower())),
                    None
                )
            t1    = pick("t1",    ex="t1ce")
            t1ce  = pick("t1ce")
            t2    = pick("t2")
            flair = pick("flair")
            seg   = pick("seg")
            if all([t1, t1ce, t2, flair, seg]):
                data_dicts.append({
                    "t1": t1, "t1ce": t1ce, "t2": t2,
                    "flair": flair, "label": seg,
                    "class_id": class_id,
                })
    print(f"[Dataset] Found {len(data_dicts)} valid patients.")
    return data_dicts
"""
Brain Tumor Radiomic Classifier — Inference Script
====================================================
Use this after training with brain_tumor_radiomic_classifier.py

Modes:
  1. SINGLE   — classify one patient folder or .nii.gz file
  2. BATCH    — classify all patients in a dataset folder
  3. EVALUATE — batch with known true labels + accuracy report

Requirements:
  pip install nibabel numpy scipy scikit-learn pandas joblib tqdm

Usage:
  Set RUN_MODE and paths at the bottom, then run.
"""

import re
import csv
import warnings
import numpy as np
import nibabel as nib

from pathlib import Path
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from scipy import ndimage
from scipy.stats import skew, kurtosis

import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  ★ EDIT THESE
# ══════════════════════════════════════════════════════════════════

MODEL_PKL = "./radiomic_output/brain_tumor_classifier.pkl"  # trained model

# Single patient
SINGLE_PATH  = "/home/23bq1a4293/Personal/project/seg_data/meningioma/BraTS-MEN-Train/BraTS-MEN-00152-000"
SINGLE_LABEL = "meningioma"   # set None if unknown

# Batch / Evaluate
BATCH_DIR    = "/home/23bq1a4293/Personal/project/seg_data/meningioma/BraTS-MEN-Train"
BATCH_TYPE   = "meningioma"       # "glioma" | "meningioma" | "pituitary" | "healthy"
BATCH_LABEL  = "meningioma"       # true label for accuracy — set None if unknown
BATCH_MAX    = None           # None = all patients
BATCH_OUTPUT = f"{BATCH_TYPE}_inference_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

RUN_MODE = "batch"           # "single" | "batch" | "evaluate_all"

# ══════════════════════════════════════════════════════════════════
#  MODALITY FILE FINDER
# ══════════════════════════════════════════════════════════════════

MODALITY_PATTERNS = {
    "t1ce_brats":        re.compile(r".*t1ce\.nii(\.gz)?$",                    re.IGNORECASE),
    "t1c_brats2024":     re.compile(r".*-t1c\.nii(\.gz)?$",                    re.IGNORECASE),
    "pituitary_cecor":   re.compile(r".*acq-CECor_T1w\.nii(\.gz)?$",           re.IGNORECASE),
    "pituitary_cesag":   re.compile(r".*acq-CESag_T1w\.nii(\.gz)?$",           re.IGNORECASE),
    "pituitary_ce3dnav": re.compile(r".*acq-CE3DNavigation_T1w\.nii(\.gz)?$",  re.IGNORECASE),
    "healthy":           re.compile(r"^healthy_\d+\.nii(\.gz)?$",              re.IGNORECASE),
}
PATTERN_PRIORITY = [
    "t1ce_brats", "t1c_brats2024",
    "pituitary_cecor", "pituitary_cesag", "pituitary_ce3dnav", "healthy",
]
SKIP_KEYWORDS = ["seg","mask","label","defacemask","seed","prediction","groundtruth","dwi","adcb"]

def find_best_modality_file(path: str) -> tuple:
    p = Path(path)
    if p.is_file():
        for key in PATTERN_PRIORITY:
            if MODALITY_PATTERNS[key].match(p.name):
                return str(p), key
        if p.name.endswith(".nii") or p.name.endswith(".nii.gz"):
            return str(p), "direct_file"
        return None, None
    if not p.is_dir():
        return None, None
    candidates = sorted(list(p.rglob("*.nii")) + list(p.rglob("*.nii.gz")))
    candidates = [f for f in candidates if not any(kw in f.name.lower() for kw in SKIP_KEYWORDS)]
    for key in PATTERN_PRIORITY:
        matches = [f for f in candidates if MODALITY_PATTERNS[key].match(f.name)]
        if matches:
            return str(matches[0]), key
    return (str(candidates[0]), "fallback") if candidates else (None, None)


# ══════════════════════════════════════════════════════════════════
#  DATASET PATH HELPERS
# ══════════════════════════════════════════════════════════════════

def get_patient_paths(dataset_dir: str, dataset_type: str) -> list:
    base = Path(dataset_dir)
    if dataset_type == "glioma":
        paths = []
        for grade in ["HGG", "LGG"]:
            gd = base / grade
            if gd.exists():
                paths += [str(d) for d in sorted(gd.iterdir()) if d.is_dir()]
        return paths or [str(d) for d in sorted(base.iterdir()) if d.is_dir()]
    elif dataset_type == "meningioma":
        return [str(d) for d in sorted(base.iterdir()) if d.is_dir()]
    elif dataset_type == "pituitary":
        paths = []
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and sub.name.startswith("sub-"):
                anat = sub / "anat"
                paths.append(str(anat) if anat.exists() else str(sub))
        return paths
    elif dataset_type == "healthy":
        return sorted([str(f) for f in base.glob("healthy_*.nii.gz")])
    else:
        raise ValueError(f"Unknown dataset_type: '{dataset_type}'")


# ══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (must match training exactly)
# ══════════════════════════════════════════════════════════════════

def extract_features(nifti_path: str) -> dict:
    """Extract the same 53 features used during training."""
    features = {}
    try:
        img  = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        affine = img.affine
        vox    = np.abs(np.diag(affine)[:3])
        shape  = data.shape

        z_row      = np.abs(affine[2, :3])
        axial_axis = int(np.argmax(z_row))
        features["axial_axis"] = float(axial_axis)

        nonzero = data[data > 0]
        if len(nonzero) < 100:
            return None

        p1  = float(np.percentile(nonzero, 1))
        p99 = float(np.percentile(nonzero, 99))
        norm = np.clip(data, p1, p99)
        norm = (norm - p1) / (p99 - p1 + 1e-8)
        nz   = norm[norm > 0]

        features["mean"]     = float(nz.mean())
        features["std"]      = float(nz.std())
        features["median"]   = float(np.median(nz))
        features["p10"]      = float(np.percentile(nz, 10))
        features["p25"]      = float(np.percentile(nz, 25))
        features["p75"]      = float(np.percentile(nz, 75))
        features["p90"]      = float(np.percentile(nz, 90))
        features["p95"]      = float(np.percentile(nz, 95))
        features["p99_val"]  = float(np.percentile(nz, 99))
        features["iqr"]      = float(np.percentile(nz,75) - np.percentile(nz,25))
        features["skewness"] = float(skew(nz))
        features["kurtosis"] = float(kurtosis(nz))
        features["cv"]       = float(nz.std() / (nz.mean() + 1e-8))
        features["nz_frac"]  = float(len(nz) / data.size)

        hi_thresh = float(np.percentile(nz, 95))
        hi_mask   = norm > hi_thresh
        features["hi_count"] = float(hi_mask.sum())
        features["hi_frac"]  = float(hi_mask.sum() / max(len(nz), 1))

        labeled, n_comp = ndimage.label(hi_mask)
        features["n_components"] = float(n_comp)

        if n_comp > 0:
            sizes       = [int((labeled == i).sum()) for i in range(1, n_comp+1)]
            largest_idx = int(np.argmax(sizes)) + 1
            main_mask   = labeled == largest_idx
            main_size   = sizes[largest_idx - 1]
            vox_vol     = float(vox[0] * vox[1] * vox[2])
            features["lesion_vol_mm3"] = float(main_size * vox_vol)

            coords   = np.argwhere(main_mask)
            centroid = coords.mean(axis=0)
            norm_c   = centroid / np.array(shape)
            features["centroid_lr"]  = float(norm_c[0])
            features["centroid_ap"]  = float(norm_c[1])
            features["centroid_si"]  = float(norm_c[axial_axis])

            extent    = coords.max(axis=0) - coords.min(axis=0)
            extent_mm = extent * vox
            features["extent_d0_mm"]  = float(extent_mm[0])
            features["extent_d1_mm"]  = float(extent_mm[1])
            features["extent_d2_mm"]  = float(extent_mm[2])
            features["max_extent_mm"] = float(extent_mm.max())

            dist_to_edge = float(min(
                norm_c[0], 1-norm_c[0],
                norm_c[1], 1-norm_c[1],
                norm_c[2], 1-norm_c[2]
            ))
            features["dist_to_edge"] = dist_to_edge

            main_vals = norm[main_mask]
            features["lesion_mean"]     = float(main_vals.mean())
            features["lesion_std"]      = float(main_vals.std())
            features["lesion_cv"]       = float(main_vals.std() / (main_vals.mean() + 1e-8))
            features["lesion_skewness"] = float(skew(main_vals))
            features["lesion_p95"]      = float(np.percentile(main_vals, 95))

            try:
                core_mask = ndimage.binary_erosion(main_mask, iterations=4)
                rim_mask  = main_mask & ~core_mask
                core_mean = float(norm[core_mask].mean()) if core_mask.any() else features["lesion_mean"]
                rim_mean  = float(norm[rim_mask].mean())  if rim_mask.any()  else features["lesion_mean"]
                features["rim_core_ratio"] = float(rim_mean / (core_mean + 1e-8))
                features["core_mean"]      = core_mean
                features["rim_mean"]       = rim_mean
            except Exception:
                features["rim_core_ratio"] = 1.0
                features["core_mean"]      = features["lesion_mean"]
                features["rim_mean"]       = features["lesion_mean"]

            bbox_vol = float(extent_mm.prod()) if extent_mm.prod() > 0 else 1.0
            features["sphericity"] = float(features["lesion_vol_mm3"] / (bbox_vol + 1e-8))
        else:
            for k in ["lesion_vol_mm3","centroid_lr","centroid_ap","centroid_si",
                      "extent_d0_mm","extent_d1_mm","extent_d2_mm","max_extent_mm",
                      "dist_to_edge","lesion_mean","lesion_std","lesion_cv",
                      "lesion_skewness","lesion_p95","rim_core_ratio",
                      "core_mean","rim_mean","sphericity"]:
                features[k] = 0.0

        mid    = shape[0] // 2
        l_mean = float(norm[:mid].mean())
        r_mean = float(norm[mid:].mean())
        features["asym_lr"] = float(abs(l_mean - r_mean) / (max(l_mean, r_mean) + 1e-8))
        features["l_mean"]  = l_mean
        features["r_mean"]  = r_mean

        sx = slice(int(shape[0]*0.40), int(shape[0]*0.60))
        sy = slice(int(shape[1]*0.35), int(shape[1]*0.65))
        sz = slice(int(shape[2]*0.10), int(shape[2]*0.35))
        sellar = norm[sx, sy, sz]
        features["sellar_mean"]  = float(sellar.mean())
        features["sellar_max"]   = float(sellar.max())
        features["sellar_ratio"] = float(sellar.max() / (hi_thresh + 1e-8))

        sup_ax = shape[axial_axis]
        if axial_axis == 2:
            sup = norm[:, :, int(sup_ax*0.55):]
        elif axial_axis == 1:
            sup = norm[:, int(sup_ax*0.55):, :]
        else:
            sup = norm[int(sup_ax*0.55):, :, :]
        features["superior_mean"] = float(sup[sup > hi_thresh].mean()) if (sup > hi_thresh).any() else 0.0
        features["superior_frac"] = float((sup > hi_thresh).sum() / max((norm > hi_thresh).sum(), 1))

        if n_comp > 0:
            gx = ndimage.sobel(norm, axis=0)
            gy = ndimage.sobel(norm, axis=1)
            gz = ndimage.sobel(norm, axis=2)
            grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            dilated   = ndimage.binary_dilation(main_mask, iterations=2)
            boundary  = dilated & ~main_mask
            features["boundary_gradient"] = float(grad_mag[boundary].mean()) if boundary.any() else 0.0
            features["interior_gradient"] = float(grad_mag[main_mask].mean()) if main_mask.any() else 0.0
            features["gradient_ratio"]    = float(
                features["boundary_gradient"] / (features["interior_gradient"] + 1e-8)
            )
        else:
            features["boundary_gradient"] = 0.0
            features["interior_gradient"] = 0.0
            features["gradient_ratio"]    = 0.0

        features["vox_x"]    = float(vox[0])
        features["vox_y"]    = float(vox[1])
        features["vox_z"]    = float(vox[2])
        features["shape_d0"] = float(shape[0])
        features["shape_d1"] = float(shape[1])
        features["shape_d2"] = float(shape[2])

    except Exception as e:
        print(f"      Feature extraction error: {e}")
        return None

    return features


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════

_trained_model = None
_label_encoder = None
_feature_cols  = None

def load_trained_model(model_pkl: str):
    global _trained_model, _label_encoder, _feature_cols

    if _trained_model is not None:
        return _trained_model, _label_encoder, _feature_cols

    if not Path(model_pkl).exists():
        raise FileNotFoundError(
            f"Model not found: {model_pkl}\n"
            "Train first by running brain_tumor_radiomic_classifier.py"
        )

    _trained_model, _label_encoder = joblib.load(model_pkl)
    print(f"  Model loaded : {model_pkl}")
    print(f"  Classes      : {list(_label_encoder.classes_)}")

    # Feature column order must match training — derive from a dummy extraction
    dummy_features = extract_features.__code__.co_consts
    # Use a fixed known order matching the training script
    _feature_cols = [
        "axial_axis","mean","std","median","p10","p25","p75","p90","p95","p99_val",
        "iqr","skewness","kurtosis","cv","nz_frac","hi_count","hi_frac","n_components",
        "lesion_vol_mm3","centroid_lr","centroid_ap","centroid_si",
        "extent_d0_mm","extent_d1_mm","extent_d2_mm","max_extent_mm","dist_to_edge",
        "lesion_mean","lesion_std","lesion_cv","lesion_skewness","lesion_p95",
        "rim_core_ratio","core_mean","rim_mean","sphericity",
        "asym_lr","l_mean","r_mean",
        "sellar_mean","sellar_max","sellar_ratio",
        "superior_mean","superior_frac",
        "boundary_gradient","interior_gradient","gradient_ratio",
        "vox_x","vox_y","vox_z","shape_d0","shape_d1","shape_d2",
    ]
    print(f"  Feature cols : {len(_feature_cols)}")
    return _trained_model, _label_encoder, _feature_cols


# ══════════════════════════════════════════════════════════════════
#  CORE PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════════

def predict_one(path: str, model, label_encoder, feature_cols: list) -> dict:
    """Predict class for one patient path (folder or .nii.gz file)."""
    import time
    patient_name = Path(path).stem if Path(path).is_file() else Path(path).name

    # Find NIfTI
    nifti_path, modality = find_best_modality_file(path)
    if nifti_path is None:
        return {
            "patient": patient_name, "nifti_file": None,
            "classification": "ERROR", "confidence": "N/A",
            "error": "No NIfTI file found",
        }

    # Extract features
    t0    = time.time()
    feats = extract_features(nifti_path)
    t_feat = time.time() - t0

    if feats is None:
        return {
            "patient": patient_name, "nifti_file": nifti_path,
            "classification": "ERROR", "confidence": "N/A",
            "error": "Feature extraction failed",
        }

    # Build feature vector in correct column order
    X_new = np.array([[feats.get(col, 0.0) for col in feature_cols]], dtype=np.float32)

    # Predict
    pred_idx   = model.predict(X_new)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Probabilities (SVM with probability=True)
    proba_dict = {}
    if hasattr(model, "predict_proba"):
        try:
            proba    = model.predict_proba(X_new)[0]
            max_prob = float(proba.max())
            confidence = "high" if max_prob > 0.70 else "medium" if max_prob > 0.50 else "low"
            for i, cls in enumerate(label_encoder.classes_):
                proba_dict[f"prob_{cls}"] = float(proba[i])
        except Exception:
            max_prob   = 1.0
            confidence = "unknown"
    else:
        confidence = "unknown"
        max_prob   = 1.0

    result = {
        "patient":        patient_name,
        "nifti_file":     nifti_path,
        "modality":       modality,
        "classification": pred_label,
        "confidence":     confidence,
        "feat_time_s":    round(t_feat, 2),
        **proba_dict,
        # Key features that drove the decision (for interpretability)
        "dist_to_edge":   round(feats.get("dist_to_edge", 0), 3),
        "rim_core_ratio": round(feats.get("rim_core_ratio", 0), 3),
        "lesion_cv":      round(feats.get("lesion_cv", 0), 3),
        "sellar_ratio":   round(feats.get("sellar_ratio", 0), 3),
        "centroid_si":    round(feats.get("centroid_si", 0), 3),
        "asym_lr":        round(feats.get("asym_lr", 0), 3),
    }
    return result


# ══════════════════════════════════════════════════════════════════
#  SINGLE PATIENT MODE
# ══════════════════════════════════════════════════════════════════

def run_single(path: str, true_label: str = None):
    model, le, feat_cols = load_trained_model(MODEL_PKL)

    name = Path(path).stem if Path(path).is_file() else Path(path).name
    print(f"\n{'='*55}")
    print(f"  Patient : {name}")
    print(f"{'='*55}")

    result = predict_one(path, model, le, feat_cols)

    print(f"\n  File           : {Path(result['nifti_file']).name if result.get('nifti_file') else 'NOT FOUND'}")
    print(f"  Modality       : {result.get('modality','N/A')}")
    print(f"  CLASSIFICATION : {result['classification'].upper()}")
    print(f"  CONFIDENCE     : {result['confidence']}")

    if true_label:
        match = "✓ CORRECT" if result["classification"] == true_label else "✗ WRONG"
        print(f"  True label     : {true_label}  →  {match}")

    # Print class probabilities
    prob_keys = [k for k in result if k.startswith("prob_")]
    if prob_keys:
        print(f"\n  Class probabilities:")
        for k in sorted(prob_keys, key=lambda x: -result[x]):
            bar_len = int(result[k] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {k[5:]:<12} {bar} {result[k]*100:5.1f}%")

    # Print key radiomic features that determined classification
    print(f"\n  Key features:")
    print(f"    dist_to_edge   : {result['dist_to_edge']:.3f}  "
          f"({'near surface → meningioma' if result['dist_to_edge'] < 0.18 else 'deep → glioma/pituitary'})")
    print(f"    rim_core_ratio : {result['rim_core_ratio']:.3f}  "
          f"({'ring pattern → glioma HGG' if result['rim_core_ratio'] > 1.6 else 'solid mass'})")
    print(f"    lesion_cv      : {result['lesion_cv']:.3f}  "
          f"({'heterogeneous' if result['lesion_cv'] > 0.25 else 'homogeneous'})")
    print(f"    sellar_ratio   : {result['sellar_ratio']:.3f}  "
          f"({'pituitary signal' if result['sellar_ratio'] > 0.85 else 'no sellar signal'})")
    print(f"    centroid_si    : {result['centroid_si']:.3f}  "
          f"({'inferior skull base → pituitary' if result['centroid_si'] < 0.30 else 'upper brain'})")
    print(f"    asym_lr        : {result['asym_lr']:.3f}  "
          f"({'asymmetric' if result['asym_lr'] > 0.05 else 'symmetric'})")
    print(f"\n  Feature extraction time: {result['feat_time_s']}s")
    print(f"{'='*55}\n")

    return result


# ══════════════════════════════════════════════════════════════════
#  BATCH / EVALUATE MODE
# ══════════════════════════════════════════════════════════════════

def run_batch(dataset_dir: str, dataset_type: str,
              true_label: str = None,
              output_csv: str = "inference_results.csv",
              max_patients: int = None) -> list:

    model, le, feat_cols = load_trained_model(MODEL_PKL)
    patient_paths = get_patient_paths(dataset_dir, dataset_type)
    if max_patients:
        patient_paths = patient_paths[:max_patients]

    print(f"\n{'='*55}")
    print(f"  BATCH INFERENCE : {dataset_type.upper()}")
    print(f"  Patients        : {len(patient_paths)}")
    print(f"  True label      : {true_label or 'unknown'}")
    print(f"  Output          : {output_csv}")
    print(f"{'='*55}\n")

    results  = []
    correct  = 0
    pred_all = []

    for i, path in enumerate(tqdm(patient_paths, desc=dataset_type)):
        name = Path(path).stem if Path(path).is_file() else Path(path).name
        result = predict_one(path, model, le, feat_cols)

        result["true_label"] = true_label or "unknown"
        result["correct"]    = (result["classification"] == true_label) if true_label else None
        if result["correct"]:
            correct += 1

        pred_all.append(result["classification"])

        tick = ""
        if true_label:
            tick = "✓" if result["correct"] else f"✗ → {result['classification']}"
        print(f"  [{i+1:3d}/{len(patient_paths)}] {name:<40} "
              f"{result['classification'].upper():<12} [{result['confidence']}]  {tick}")

        results.append(result)

    # Save CSV
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Summary
    pred_counts = dict(Counter(pred_all))
    print(f"\n{'='*55}")
    print(f"  INFERENCE SUMMARY")
    print(f"{'='*55}")
    print(f"  Total patients   : {len(results)}")
    print(f"  Predictions      : {pred_counts}")
    if true_label:
        acc = correct / len(results) * 100 if results else 0
        print(f"  Correct          : {correct}/{len(results)} = {acc:.1f}%")
    print(f"  Saved to         : {output_csv}")
    print(f"{'='*55}\n")

    return results


# ══════════════════════════════════════════════════════════════════
#  EVALUATE ALL 4 CLASSES
# ══════════════════════════════════════════════════════════════════

DATA_ROOTS = {
    "glioma":     "/home/23bq1a4293/Personal/project/seg_data/glioma/MICCAI_BraTS_2019_Data_Training",
    "meningioma": "/home/23bq1a4293/Personal/project/seg_data/meningioma/BraTS-MEN-Train",
    "pituitary":  "/home/23bq1a4293/Personal/project/seg_data/pitutary/pituitary_tumor_dataset",
    "healthy":    "/home/23bq1a4293/Personal/project/seg_data/healthy",
}

def run_evaluate_all(max_per_class: int = 20):
    """Evaluate all 4 classes and print a combined report."""
    from sklearn.metrics import classification_report, confusion_matrix

    model, le, feat_cols = load_trained_model(MODEL_PKL)

    print(f"\n{'='*55}")
    print(f"  FULL EVALUATION — ALL 4 CLASSES")
    print(f"  Patients per class: {max_per_class}")
    print(f"{'='*55}\n")

    all_true = []
    all_pred = []
    all_results = []

    for label, data_root in DATA_ROOTS.items():
        if not Path(data_root).exists():
            print(f"  SKIP: {label} — path not found")
            continue

        paths = get_patient_paths(data_root, label)[:max_per_class]
        print(f"\n  [{label.upper()}] — {len(paths)} patients")

        for path in tqdm(paths, desc=f"  {label}", leave=False):
            result = predict_one(path, model, le, feat_cols)
            result["true_label"] = label
            result["correct"]    = result["classification"] == label
            all_true.append(label)
            all_pred.append(result["classification"])
            all_results.append(result)

            tick = "✓" if result["correct"] else f"✗ → {result['classification']}"
            name = Path(path).stem if Path(path).is_file() else Path(path).name
            print(f"    {name:<40} {result['classification'].upper():<12} {tick}")

    # Report
    print(f"\n{'='*55}")
    print(f"  CLASSIFICATION REPORT")
    print(f"{'='*55}")
    classes = sorted(set(all_true))
    print(classification_report(all_true, all_pred, target_names=classes, digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=classes)
    print(f"  Confusion matrix (rows=true, cols=pred):")
    header = f"  {'':>12}" + "".join(f"{c[:6]:>8}" for c in classes)
    print(header)
    for i, row_label in enumerate(classes):
        row = f"  {row_label:>12}" + "".join(f"{v:>8}" for v in cm[i])
        print(row)

    acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true) * 100
    print(f"\n  Overall accuracy : {acc:.1f}%")

    # Save
    out_csv = f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  Results saved    : {out_csv}")
    print(f"{'='*55}\n")

    return all_results


# ══════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if RUN_MODE == "single":
        run_single(SINGLE_PATH, SINGLE_LABEL)

    elif RUN_MODE == "batch":
        run_batch(
            dataset_dir  = BATCH_DIR,
            dataset_type = BATCH_TYPE,
            true_label   = BATCH_LABEL,
            output_csv   = BATCH_OUTPUT,
            max_patients = BATCH_MAX,
        )

    elif RUN_MODE == "evaluate_all":
        run_evaluate_all(max_per_class=20)   # 20 per class = 80 patients total

    else:
        print(f"Unknown RUN_MODE: '{RUN_MODE}'. Use: single | batch | evaluate_all")
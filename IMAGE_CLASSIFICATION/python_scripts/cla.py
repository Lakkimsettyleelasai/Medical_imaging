"""
Brain Tumor Classification — Radiomic Features + Clustering + ML
=================================================================
No VLM needed. Extracts rich 3D features from NIfTI volumes,
clusters them, then trains a supervised classifier on top.

Pipeline:
  1. Extract radiomic features from each NIfTI (intensity, texture, shape)
  2. Reduce dimensions with PCA
  3. Cluster with KMeans to find natural groupings
  4. Train RandomForest / XGBoost classifier on features
  5. Evaluate with cross-validation + confusion matrix

Datasets:
  Glioma      : *_t1ce.nii          (BraTS 2019)
  Meningioma  : *-t1c.nii.gz        (BraTS 2024)
  Pituitary   : *acq-CECor_T1w*     (BIDS)
  Healthy     : healthy_XXX.nii.gz  (flat)

Requirements:
  pip install nibabel numpy scipy scikit-learn tqdm matplotlib
              SimpleITK pandas joblib
"""

import re
import csv
import warnings
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from pathlib import Path
from collections import Counter
from datetime import datetime
from tqdm import tqdm

import pandas as pd
from scipy import ndimage
from scipy.stats import skew, kurtosis

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.pipeline import Pipeline
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  ★ EDIT THESE PATHS
# ══════════════════════════════════════════════════════════════════

DATA_ROOTS = {
    "glioma":      "/home/23bq1a4293/Personal/project/seg_data/glioma/MICCAI_BraTS_2019_Data_Training",
    "meningioma":  "/home/23bq1a4293/Personal/project/seg_data/meningioma/BraTS-MEN-Train",
    "pituitary":   "/home/23bq1a4293/Personal/project/seg_data/pitutary/pituitary_tumor_dataset",
    "healthy":     "/home/23bq1a4293/Personal/project/seg_data/healthy",
}

OUTPUT_DIR    = "./radiomic_output"
MAX_PER_CLASS = 100    # limit patients per class (set None for all)
N_JOBS        = 4      # parallel workers for feature extraction
RANDOM_STATE  = 42

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
SKIP_KEYWORDS = [
    "seg", "mask", "label", "defacemask", "seed",
    "prediction", "groundtruth", "dwi", "adcb"
]

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
    candidates = [f for f in candidates
                  if not any(kw in f.name.lower() for kw in SKIP_KEYWORDS)]
    for key in PATTERN_PRIORITY:
        matches = [f for f in candidates if MODALITY_PATTERNS[key].match(f.name)]
        if matches:
            return str(matches[0]), key
    return (str(candidates[0]), "fallback") if candidates else (None, None)


# ══════════════════════════════════════════════════════════════════
#  DATASET PATH COLLECTORS
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
        raise ValueError(f"Unknown: {dataset_type}")


# ══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
#  Extracts ~60 radiomic features per volume covering:
#   - Intensity statistics (global + non-zero)
#   - Enhancement metrics (high intensity cluster properties)
#   - Spatial location (centroid, proximity to skull surface)
#   - Texture (GLCM-inspired: contrast, energy, homogeneity)
#   - Shape (extent, sphericity proxy, ring detection)
#   - Symmetry (left-right hemisphere comparison)
#   - Sellar region probe (for pituitary detection)
# ══════════════════════════════════════════════════════════════════

def extract_features(nifti_path: str) -> dict:
    """Extract ~60 radiomic features from a NIfTI volume."""
    features = {}

    try:
        img  = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        affine = img.affine
        vox    = np.abs(np.diag(affine)[:3])
        shape  = data.shape

        # ── Detect axial axis ──────────────────────────────────────
        z_row      = np.abs(affine[2, :3])
        axial_axis = int(np.argmax(z_row))
        features["axial_axis"] = float(axial_axis)

        # ── Normalize ──────────────────────────────────────────────
        nonzero = data[data > 0]
        if len(nonzero) < 100:
            return None   # empty/corrupt volume

        p1  = float(np.percentile(nonzero, 1))
        p99 = float(np.percentile(nonzero, 99))
        norm = np.clip(data, p1, p99)
        norm = (norm - p1) / (p99 - p1 + 1e-8)

        nz = norm[norm > 0]

        # ── Global intensity statistics ────────────────────────────
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
        features["cv"]       = float(nz.std() / (nz.mean() + 1e-8))  # coeff variation
        features["nz_frac"]  = float(len(nz) / data.size)

        # ── Enhancement cluster (top 5% intensity) ────────────────
        hi_thresh = float(np.percentile(nz, 95))
        hi_mask   = norm > hi_thresh
        features["hi_count"] = float(hi_mask.sum())
        features["hi_frac"]  = float(hi_mask.sum() / max(len(nz), 1))

        # Largest connected component of high-intensity region
        labeled, n_comp = ndimage.label(hi_mask)
        features["n_components"] = float(n_comp)

        if n_comp > 0:
            sizes       = [int((labeled == i).sum()) for i in range(1, n_comp+1)]
            largest_idx = int(np.argmax(sizes)) + 1
            main_mask   = labeled == largest_idx
            main_size   = sizes[largest_idx - 1]
            vox_vol     = float(vox[0] * vox[1] * vox[2])
            features["lesion_vol_mm3"] = float(main_size * vox_vol)

            # Centroid (normalised 0-1)
            coords   = np.argwhere(main_mask)
            centroid = coords.mean(axis=0)
            norm_c   = centroid / np.array(shape)
            features["centroid_lr"]  = float(norm_c[0])
            features["centroid_ap"]  = float(norm_c[1])
            features["centroid_si"]  = float(norm_c[axial_axis])

            # Extent (mm)
            extent    = coords.max(axis=0) - coords.min(axis=0)
            extent_mm = extent * vox
            features["extent_d0_mm"] = float(extent_mm[0])
            features["extent_d1_mm"] = float(extent_mm[1])
            features["extent_d2_mm"] = float(extent_mm[2])
            features["max_extent_mm"]= float(extent_mm.max())

            # Proximity to skull surface (0=edge, 0.5=center)
            dist_to_edge = float(min(
                norm_c[0], 1-norm_c[0],
                norm_c[1], 1-norm_c[1],
                norm_c[2], 1-norm_c[2]
            ))
            features["dist_to_edge"] = dist_to_edge

            # Enhancement statistics within lesion
            main_vals = norm[main_mask]
            features["lesion_mean"]     = float(main_vals.mean())
            features["lesion_std"]      = float(main_vals.std())
            features["lesion_cv"]       = float(main_vals.std() / (main_vals.mean() + 1e-8))
            features["lesion_skewness"] = float(skew(main_vals))
            features["lesion_p95"]      = float(np.percentile(main_vals, 95))

            # Ring detection: erode mask, compare core vs rim intensity
            try:
                core_mask  = ndimage.binary_erosion(main_mask, iterations=4)
                rim_mask   = main_mask & ~core_mask
                core_mean  = float(norm[core_mask].mean()) if core_mask.any() else features["lesion_mean"]
                rim_mean   = float(norm[rim_mask].mean())  if rim_mask.any()  else features["lesion_mean"]
                features["rim_core_ratio"] = float(rim_mean / (core_mean + 1e-8))
                features["core_mean"]      = core_mean
                features["rim_mean"]       = rim_mean
            except Exception:
                features["rim_core_ratio"] = 1.0
                features["core_mean"]      = features["lesion_mean"]
                features["rim_mean"]       = features["lesion_mean"]

            # Sphericity proxy: volume / bounding box volume
            bbox_vol = float(extent_mm.prod()) if extent_mm.prod() > 0 else 1.0
            features["sphericity"] = float(features["lesion_vol_mm3"] / (bbox_vol + 1e-8))

        else:
            # No lesion found — fill with zeros
            for k in ["lesion_vol_mm3","centroid_lr","centroid_ap","centroid_si",
                      "extent_d0_mm","extent_d1_mm","extent_d2_mm","max_extent_mm",
                      "dist_to_edge","lesion_mean","lesion_std","lesion_cv",
                      "lesion_skewness","lesion_p95","rim_core_ratio",
                      "core_mean","rim_mean","sphericity"]:
                features[k] = 0.0

        # ── Symmetry (left-right hemisphere) ──────────────────────
        mid    = shape[0] // 2
        l_mean = float(norm[:mid].mean())
        r_mean = float(norm[mid:].mean())
        features["asym_lr"] = float(abs(l_mean - r_mean) / (max(l_mean, r_mean) + 1e-8))
        features["l_mean"]  = l_mean
        features["r_mean"]  = r_mean

        # ── Sellar region probe (pituitary detection) ─────────────
        # Central inferior 15-35% of the axial axis = sella turcica region
        sx = slice(int(shape[0]*0.40), int(shape[0]*0.60))
        sy = slice(int(shape[1]*0.35), int(shape[1]*0.65))
        sz = slice(int(shape[2]*0.10), int(shape[2]*0.35))
        sellar       = norm[sx, sy, sz]
        features["sellar_mean"]  = float(sellar.mean())
        features["sellar_max"]   = float(sellar.max())
        features["sellar_ratio"] = float(sellar.max() / (hi_thresh + 1e-8))

        # ── Superior region probe (glioma — often in upper hemispheres) ──
        sup_ax = norm.shape[axial_axis]
        if axial_axis == 2:
            sup = norm[:, :, int(sup_ax*0.55):]
        elif axial_axis == 1:
            sup = norm[:, int(sup_ax*0.55):, :]
        else:
            sup = norm[int(sup_ax*0.55):, :, :]
        features["superior_mean"] = float(sup[sup > hi_thresh].mean()) if (sup > hi_thresh).any() else 0.0
        features["superior_frac"] = float((sup > hi_thresh).sum() / max((norm > hi_thresh).sum(), 1))

        # ── Simple texture: gradient magnitude (edge sharpness) ───
        # Sharp borders (meningioma) → high gradient at lesion boundary
        # Infiltrative borders (glioma) → gradual gradient
        if n_comp > 0:
            gx = ndimage.sobel(norm, axis=0)
            gy = ndimage.sobel(norm, axis=1)
            gz = ndimage.sobel(norm, axis=2)
            grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            # Gradient at lesion boundary (dilated - eroded)
            dilated  = ndimage.binary_dilation(main_mask, iterations=2)
            boundary = dilated & ~main_mask
            features["boundary_gradient"] = float(grad_mag[boundary].mean()) if boundary.any() else 0.0
            features["interior_gradient"] = float(grad_mag[main_mask].mean()) if main_mask.any() else 0.0
            features["gradient_ratio"]    = float(features["boundary_gradient"] /
                                                   (features["interior_gradient"] + 1e-8))
        else:
            features["boundary_gradient"] = 0.0
            features["interior_gradient"] = 0.0
            features["gradient_ratio"]    = 0.0

        # ── Volume metadata ────────────────────────────────────────
        features["vox_x"]   = float(vox[0])
        features["vox_y"]   = float(vox[1])
        features["vox_z"]   = float(vox[2])
        features["shape_d0"]= float(shape[0])
        features["shape_d1"]= float(shape[1])
        features["shape_d2"]= float(shape[2])

    except Exception as e:
        print(f"      Feature extraction error: {e}")
        return None

    return features


# ══════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ══════════════════════════════════════════════════════════════════

def load_all_features(max_per_class: int = None) -> pd.DataFrame:
    """Extract features from all datasets and return a DataFrame."""
    all_rows = []

    for label, data_root in DATA_ROOTS.items():
        if not Path(data_root).exists():
            print(f"  WARNING: {data_root} not found — skipping {label}")
            continue

        paths = get_patient_paths(data_root, label)
        if max_per_class:
            paths = paths[:max_per_class]

        print(f"\n  {label.upper()} : {len(paths)} patients")

        for path in tqdm(paths, desc=f"  {label}", leave=False):
            patient_name = Path(path).stem if Path(path).is_file() else Path(path).name
            nifti_path, modality = find_best_modality_file(path)

            if nifti_path is None:
                continue

            feats = extract_features(nifti_path)
            if feats is None:
                continue

            feats["patient"]  = patient_name
            feats["label"]    = label
            feats["modality"] = modality
            feats["path"]     = str(path)
            all_rows.append(feats)

    df = pd.DataFrame(all_rows)
    print(f"\n  Total samples: {len(df)}")
    print(f"  Class counts : {dict(df['label'].value_counts())}")
    return df


# ══════════════════════════════════════════════════════════════════
#  CLUSTERING ANALYSIS
# ══════════════════════════════════════════════════════════════════

def run_clustering(X_scaled: np.ndarray, y: np.ndarray,
                   label_names: list, pca_components: int = 20):
    """
    PCA + KMeans clustering.
    Shows how well natural clusters align with true labels.
    """
    print(f"\n{'='*60}")
    print(f"  CLUSTERING ANALYSIS")
    print(f"{'='*60}")

    # PCA
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA {pca_components} components explain: {explained[-1]*100:.1f}% variance")

    # KMeans with k=4 (one per class)
    n_classes = len(set(y))
    km = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=20)
    cluster_labels = km.fit_predict(X_pca)

    # Map clusters to class labels by majority vote
    cluster_to_class = {}
    for c in range(n_classes):
        mask = cluster_labels == c
        if mask.any():
            majority = Counter(y[mask]).most_common(1)[0][0]
            cluster_to_class[c] = majority

    predicted = np.array([cluster_to_class.get(c, 0) for c in cluster_labels])
    acc = accuracy_score(y, predicted)
    print(f"  KMeans clustering accuracy (majority vote): {acc*100:.1f}%")

    print(f"\n  Cluster composition:")
    for c in range(n_classes):
        mask   = cluster_labels == c
        counts = Counter(y[mask])
        mapped = label_names[cluster_to_class.get(c, 0)]
        print(f"    Cluster {c} → mapped to '{mapped}': "
              + ", ".join(f"{label_names[k]}:{v}" for k, v in counts.items()))

    return X_pca, pca, cluster_labels


# ══════════════════════════════════════════════════════════════════
#  SUPERVISED CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def run_classification(X: np.ndarray, y: np.ndarray,
                       label_names: list, feature_names: list):
    """
    Train and cross-validate multiple classifiers.
    Returns the best model.
    """
    print(f"\n{'='*60}")
    print(f"  SUPERVISED CLASSIFICATION")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    classifiers = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=RANDOM_STATE)),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=RANDOM_STATE)),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                random_state=RANDOM_STATE
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=RANDOM_STATE)),
            ("clf",    SVC(
                kernel="rbf", C=10, gamma="scale",
                class_weight="balanced", probability=True,
                random_state=RANDOM_STATE
            )),
        ]),
    }

    results = {}
    best_model = None
    best_score = 0.0

    for name, pipe in classifiers.items():
        scores = cross_val_score(pipe, X, y, cv=cv,
                                 scoring="balanced_accuracy", n_jobs=-1)
        mean_acc = scores.mean()
        results[name] = scores
        print(f"\n  {name}:")
        print(f"    CV balanced accuracy: {mean_acc*100:.1f}% ± {scores.std()*100:.1f}%")
        print(f"    Per-fold: {[f'{s*100:.1f}%' for s in scores]}")

        if mean_acc > best_score:
            best_score = mean_acc
            best_model = (name, pipe)

    # Train best model on full data and show feature importance
    print(f"\n  Best model: {best_model[0]} ({best_score*100:.1f}%)")
    best_model[1].fit(X, y)

    # Feature importance (RandomForest / GBM only)
    clf = best_model[1].named_steps.get("clf")
    if hasattr(clf, "feature_importances_"):
        pca_obj   = best_model[1].named_steps.get("pca")
        # Since PCA transforms features, show top original features
        # by reconstructing importance via PCA components
        pca_comp  = pca_obj.components_          # (n_pca, n_features)
        imp       = clf.feature_importances_     # (n_pca,)
        orig_imp  = np.abs(pca_comp * imp[:, None]).sum(axis=0)
        top_idx   = np.argsort(orig_imp)[::-1][:15]
        print(f"\n  Top 15 most important features:")
        for i, idx in enumerate(top_idx):
            if idx < len(feature_names):
                print(f"    {i+1:2d}. {feature_names[idx]:<30} {orig_imp[idx]:.4f}")

    return best_model[1], results


# ══════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX + PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(model, X: np.ndarray, y: np.ndarray,
                          label_names: list, save_path: str):
    """Full confusion matrix on all data."""
    from sklearn.model_selection import cross_val_predict
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred  = cross_val_predict(model, X, y, cv=cv)
    cm      = confusion_matrix(y, y_pred)
    acc     = accuracy_score(y, y_pred)

    print(f"\n  Classification Report (5-fold CV predictions):")
    print(classification_report(y, y_pred,
                                target_names=label_names, digits=3))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion Matrix (5-fold CV) — Accuracy: {acc*100:.1f}%"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")
    return acc


def plot_pca_scatter(X_pca: np.ndarray, y: np.ndarray,
                     label_names: list, save_path: str):
    """2D PCA scatter plot coloured by class."""
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(label_names):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[i], label=label, alpha=0.6, s=30, edgecolors="none")
    ax.set(xlabel="PC1", ylabel="PC2",
           title="PCA of Radiomic Features")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PCA scatter saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
#  PREDICT NEW PATIENT  (inference mode)
# ══════════════════════════════════════════════════════════════════

def predict_patient(path: str, model_path: str = None,
                    trained_model=None, label_encoder=None) -> dict:
    """
    Predict class for a single new patient using saved model.
    """
    if trained_model is None:
        if model_path is None:
            raise ValueError("Provide either trained_model or model_path")
        trained_model, label_encoder = joblib.load(model_path)

    patient_name = Path(path).stem if Path(path).is_file() else Path(path).name
    nifti_path, modality = find_best_modality_file(path)

    if nifti_path is None:
        return {"patient": patient_name, "error": "No NIfTI found"}

    feats = extract_features(nifti_path)
    if feats is None:
        return {"patient": patient_name, "error": "Feature extraction failed"}

    # Get feature names from saved model
    feature_cols = [k for k in feats.keys()
                    if k not in ("patient","label","modality","path","axial_axis")]
    X_new = np.array([[feats.get(c, 0.0) for c in feature_cols]])

    pred_idx   = trained_model.predict(X_new)[0]
    pred_proba = trained_model.predict_proba(X_new)[0] if hasattr(trained_model, "predict_proba") else None
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    confidence = "high" if (pred_proba is not None and pred_proba.max() > 0.7) else \
                 "medium" if (pred_proba is not None and pred_proba.max() > 0.5) else "low"

    result = {
        "patient":        patient_name,
        "classification": pred_label,
        "confidence":     confidence,
        "modality":       modality,
    }
    if pred_proba is not None:
        for i, lbl in enumerate(label_encoder.classes_):
            result[f"prob_{lbl}"] = float(pred_proba[i])

    return result


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_full_pipeline():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BRAIN TUMOR RADIOMIC CLASSIFICATION PIPELINE")
    print(f"{'='*60}")
    print(f"  Output directory : {out.resolve()}")
    print(f"  Max per class    : {MAX_PER_CLASS}")

    # ── Step 1: Extract features ───────────────────────────────────
    print(f"\n[STEP 1] Extracting radiomic features...")
    df = load_all_features(max_per_class=MAX_PER_CLASS)

    if len(df) < 20:
        print("ERROR: Not enough samples extracted. Check DATA_ROOTS paths.")
        return

    # Save raw features
    feat_csv = str(out / "radiomic_features.csv")
    df.to_csv(feat_csv, index=False)
    print(f"  Features saved: {feat_csv}")

    # ── Prepare X, y ───────────────────────────────────────────────
    meta_cols    = ["patient", "label", "modality", "path"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X            = df[feature_cols].fillna(0).values.astype(np.float32)
    le           = LabelEncoder()
    y            = le.fit_transform(df["label"].values)
    label_names  = list(le.classes_)

    print(f"\n  Features shape  : {X.shape}")
    print(f"  Classes         : {label_names}")
    print(f"  Class counts    : {dict(Counter(df['label'].values))}")

    # ── Step 2: Scale ──────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 3: Clustering ─────────────────────────────────────────
    print(f"\n[STEP 2] Running clustering analysis...")
    X_pca, pca, cluster_labels = run_clustering(
        X_scaled, y, label_names, pca_components=min(20, X.shape[1]-1)
    )
    plot_pca_scatter(X_pca, y, label_names, str(out / "pca_scatter.png"))

    # ── Step 4: Supervised classification ─────────────────────────
    print(f"\n[STEP 3] Training classifiers...")
    best_model, cv_results = run_classification(
        X, y, label_names, feature_cols
    )

    # ── Step 5: Confusion matrix ───────────────────────────────────
    print(f"\n[STEP 4] Generating confusion matrix...")
    acc = plot_confusion_matrix(
        best_model, X, y, label_names,
        str(out / "confusion_matrix.png")
    )

    # ── Step 6: Save model ─────────────────────────────────────────
    model_save_path = str(out / "brain_tumor_classifier.pkl")
    joblib.dump((best_model, le), model_save_path)
    print(f"\n  Model saved: {model_save_path}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Samples         : {len(df)}")
    print(f"  Features        : {X.shape[1]}")
    print(f"  Overall accuracy: {acc*100:.1f}%")
    print(f"\n  Output files:")
    print(f"    {feat_csv}")
    print(f"    {out}/confusion_matrix.png")
    print(f"    {out}/pca_scatter.png")
    print(f"    {model_save_path}")
    print(f"{'='*60}\n")

    return best_model, le, df


# ══════════════════════════════════════════════════════════════════
#  SINGLE PATIENT PREDICTION  (after training)
# ══════════════════════════════════════════════════════════════════

def predict_single(patient_path: str,
                   model_pkl: str = "./radiomic_output/brain_tumor_classifier.pkl"):
    """Predict class for one patient using the trained model."""
    print(f"\nPredicting: {Path(patient_path).name}")
    result = predict_patient(patient_path, model_path=model_pkl)
    print(f"  Classification : {result.get('classification','ERROR').upper()}")
    print(f"  Confidence     : {result.get('confidence','N/A')}")
    for k, v in result.items():
        if k.startswith("prob_"):
            print(f"  {k:<25}: {v:.3f}")
    return result


# ══════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1: Train on all data
    best_model, le, df = run_full_pipeline()

    # Step 2: Predict a new patient (after training)
    # predict_single(
    #     "/path/to/new/patient/folder",
    #     model_pkl="./radiomic_output/brain_tumor_classifier.pkl"
    # )
"""
Brain Tumor Segmentation Pipeline — FastAPI Backend
====================================================
Endpoints:
  POST /classify      — detect tumor type from best slice
  POST /segment       — full segmentation based on tumor type
  POST /pipeline      — end-to-end: classify → segment → visualize
  GET  /health        — health check

Install:
  pip install fastapi uvicorn python-multipart nibabel scipy torch monai

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import io
import json
import base64
import tempfile
import warnings
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy.ndimage import zoom
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =====================================================================
# CONFIG — Update these paths
# =====================================================================
PATHS = {
    "pituitary_results": "/home/23bq1a4293/Personal/project/nnunet/nnUNet_results/Dataset001_Pituitary/nnUNetTrainer__nnUNetPlans__2d",
    "meningioma_model":  "/home/23bq1a4293/Personal/project/models/Segmentation/men/best_model_smoothed_4class.pth",
    "glioma_model":      "/home/23bq1a4293/Personal/project/models/Segmentation/glioma/best_segresnet_brats.pth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUMOR_LABELS = {
    "pituitary": {
        1: "tumor", 2: "normal_gland",
        3: "ICA_left", 4: "ICA_right", 5: "optic_chiasm"
    },
    "meningioma": {
        1: "necrotic_core", 2: "peritumoral_edema", 3: "enhancing_tumor"
    },
    "glioma": {
        1: "necrotic_core", 2: "peritumoral_edema", 3: "enhancing_tumor"
    },
}

TUMOR_COLORS = {
    "pituitary": {1:[1,0,0], 2:[0,1,0], 3:[0,0.4,1], 4:[1,1,0], 5:[1,0,1]},
    "meningioma": {1:[1,0,0], 2:[0,1,0], 3:[1,1,0]},
    "glioma":     {1:[1,0,0], 2:[0,1,0], 3:[1,1,0]},
}

# =====================================================================
# MODEL LOADING
# =====================================================================
_models = {}


def get_pituitary_model(fold: int):
    key = f"pituitary_fold{fold}"
    if key not in _models:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        model = PlainConvUNet(
            input_channels=1, n_stages=8,
            features_per_stage=(32,64,128,256,512,512,512,512),
            conv_op=torch.nn.Conv2d,
            kernel_sizes=[[3,3]]*8,
            strides=[[1,1],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2]],
            n_conv_per_stage=[2]*8, num_classes=6,
            n_conv_per_stage_decoder=[2]*7, conv_bias=True,
            norm_op=torch.nn.InstanceNorm2d,
            norm_op_kwargs={'eps':1e-5,'affine':True},
            dropout_op=None, dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={'inplace':True},
            deep_supervision=False,
        ).to(DEVICE)
        ckpt_path = Path(PATHS["pituitary_results"]) / f"fold_{fold}" / "checkpoint_best.pth"
        ckpt  = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        state = {k.replace("network.", ""): v for k, v in ckpt['network_weights'].items()}
        model.load_state_dict(state)
        model.eval()
        _models[key] = model
        print(f"✅ Pituitary fold {fold} loaded")
    return _models[key]


def get_meningioma_model():
    if "meningioma" not in _models:
        from monai.networks.nets import SegResNet
        model = SegResNet(
            spatial_dims=3, in_channels=4,
            out_channels=4, init_filters=32
        ).to(DEVICE)
        model.load_state_dict(
            torch.load(PATHS["meningioma_model"], map_location=DEVICE, weights_only=True)
        )
        model.eval()
        _models["meningioma"] = model
        print("✅ Meningioma model loaded")
    return _models["meningioma"]


def get_glioma_model():
    if "glioma" not in _models:
        from monai.networks.nets import SegResNet
        model = SegResNet(
            spatial_dims=3, in_channels=4,
            out_channels=3, init_filters=32
        ).to(DEVICE)
        ckpt  = torch.load(PATHS["glioma_model"], map_location=DEVICE, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        model.eval()
        _models["glioma"] = model
        print("✅ Glioma model loaded")
    return _models["glioma"]


# =====================================================================
# UTILS
# =====================================================================
def normalize(arr: np.ndarray) -> np.ndarray:
    return ((arr - arr.mean()) / (arr.std() + 1e-8)).astype(np.float32)


def load_nifti_from_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(data)
        tmp = f.name
    img = nib.load(tmp)
    arr = img.get_fdata(dtype=np.float32)
    os.unlink(tmp)
    return arr, img


def classify_tumor(arr: np.ndarray) -> dict:
    shape = arr.shape
    # Pituitary: thin coronal axis (one dim ≤ 20)
    if arr.ndim == 3 and min(shape) <= 20:
        return {
            "tumor_type": "pituitary",
            "confidence": 0.90,
            "reason": f"Thin coronal axis (min_dim={min(shape)})",
            "shape": list(shape),
        }
    # 4-channel → meningioma or glioma
    if arr.ndim == 4 or (arr.ndim == 3 and shape[0] == 4):
        return {
            "tumor_type": "meningioma",
            "confidence": 0.75,
            "reason": "4-channel BraTS-style input",
            "shape": list(shape),
        }
    # Large single channel → glioma
    if arr.ndim == 3 and np.prod(shape) > 5_000_000:
        return {
            "tumor_type": "glioma",
            "confidence": 0.70,
            "reason": "Large 3D single-channel volume",
            "shape": list(shape),
        }
    return {
        "tumor_type": "unknown",
        "confidence": 0.0,
        "reason": f"Cannot classify shape {shape}",
        "shape": list(shape),
    }


# =====================================================================
# INFERENCE
# =====================================================================
def run_pituitary_inference(arr: np.ndarray) -> np.ndarray:
    arr_n = normalize(arr)
    H, D, W = arr_n.shape
    if H != 512 or W != 512:
        arr_n = zoom(arr_n, (512/H, 1.0, 512/W), order=1)
    m0 = get_pituitary_model(0)
    m1 = get_pituitary_model(1)
    pred = np.zeros(arr_n.shape, dtype=np.uint8)
    with torch.no_grad():
        for y in range(arr_n.shape[1]):
            sl = torch.from_numpy(arr_n[:,y,:]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            o0 = F.softmax(m0(sl), dim=1)
            o1 = F.softmax(m1(sl), dim=1)
            pred[:,y,:] = torch.argmax((o0+o1)/2, dim=1).squeeze(0).cpu().numpy()
    return pred


def run_meningioma_inference(arr: np.ndarray) -> np.ndarray:
    from monai.inferers import sliding_window_inference
    if arr.ndim == 3:
        arr = arr[np.newaxis]
    normed = np.stack([normalize(arr[i]) for i in range(arr.shape[0])], axis=0)
    tensor = torch.from_numpy(normed).float().unsqueeze(0).to(DEVICE)
    model  = get_meningioma_model()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = sliding_window_inference(
                tensor, (192,192,128), 2, model, overlap=0.5)
    return torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def run_glioma_inference(arr: np.ndarray) -> np.ndarray:
    from monai.inferers import sliding_window_inference
    if arr.ndim == 3:
        arr = arr[np.newaxis]
    normed = np.stack([normalize(arr[i]) for i in range(arr.shape[0])], axis=0)
    tensor = torch.from_numpy(normed).float().unsqueeze(0).to(DEVICE)
    model  = get_glioma_model()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = sliding_window_inference(
                tensor, (128,128,128), 2, model, overlap=0.5)
    return torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


# =====================================================================
# VISUALIZATION
# =====================================================================
def create_visualization(image: np.ndarray, pred: np.ndarray,
                         tumor_type: str, pid: str = "patient") -> str:
    colors = TUMOR_COLORS.get(tumor_type, TUMOR_COLORS["pituitary"])
    labels = TUMOR_LABELS.get(tumor_type, {})

    # Find best slice
    thin_axis = int(np.argmin(image.shape)) if image.ndim == 3 else 2
    fg = (pred > 0)
    axes_to_sum = tuple(i for i in range(3) if i != thin_axis)
    fg_counts   = fg.sum(axis=axes_to_sum)
    best_idx    = int(np.argmax(fg_counts)) if fg_counts.max() > 0 \
                  else image.shape[thin_axis]//2

    slices = [slice(None)]*3
    slices[thin_axis] = best_idx
    img_sl  = image[tuple(slices)]
    pred_sl = pred[tuple(slices)]

    # Normalize
    mn, mx = img_sl.min(), img_sl.max()
    img_n  = (img_sl - mn) / (mx - mn + 1e-8)

    # Overlay
    rgb = np.stack([img_n]*3, axis=-1).copy()
    for cls, col in colors.items():
        mask = pred_sl == cls
        if not mask.any(): continue
        for c in range(3):
            rgb[:,:,c] = np.where(mask, rgb[:,:,c]*0.4+col[c]*0.6, rgb[:,:,c])
    rgb = np.clip(rgb, 0, 1)

    # Mask only
    mask_rgb = np.zeros((*pred_sl.shape, 3))
    for cls, col in colors.items():
        mask = pred_sl == cls
        if not mask.any(): continue
        for c in range(3):
            mask_rgb[:,:,c] = np.where(mask, col[c], mask_rgb[:,:,c])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(axes,
        [img_n, rgb, mask_rgb],
        ["Input MRI", f"Segmentation ({tumor_type})", "Mask Only"]):
        ax.imshow(data.T if data.ndim == 2 else data.transpose(1,0,2),
                  cmap="gray" if data.ndim == 2 else None, origin="lower")
        ax.set_title(title, fontsize=11); ax.axis("off")

    detected = [labels.get(int(l), f"cls{l}") for l in np.unique(pred_sl) if l > 0]
    patches  = [mpatches.Patch(color=colors[i], label=labels.get(i,f"cls{i}"))
                for i in colors if i in np.unique(pred_sl)]
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=len(patches),
                   fontsize=10, bbox_to_anchor=(0.5,-0.05))

    plt.suptitle(
        f"{tumor_type.capitalize()} | {pid} | "
        f"Detected: {', '.join(detected) if detected else 'none'}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def compute_stats(pred: np.ndarray, tumor_type: str) -> dict:
    labels = TUMOR_LABELS.get(tumor_type, {})
    stats  = {"total_foreground_voxels": int((pred > 0).sum())}
    for cls, name in labels.items():
        count = int((pred == cls).sum())
        stats[name] = {"voxels": count, "present": count > 0}
    return stats


# =====================================================================
# FASTAPI APP
# =====================================================================
app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="Automated segmentation for Pituitary, Meningioma and Glioma",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": list(_models.keys()),
        "supported_tumor_types": ["pituitary", "meningioma", "glioma"],
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Upload NIfTI → get tumor type classification."""
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "Only .nii/.nii.gz supported")
    data = await file.read()
    arr, _ = load_nifti_from_bytes(data)
    result = classify_tumor(arr)
    return JSONResponse({"filename": file.filename, **result})


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    tumor_type: Optional[str] = None,
):
    """
    Upload NIfTI → get segmentation mask + visualization.
    tumor_type: pituitary | meningioma | glioma (auto-detected if not set)
    """
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "Only .nii/.nii.gz supported")

    data = await file.read()
    arr, _ = load_nifti_from_bytes(data)

    if tumor_type is None:
        cls = classify_tumor(arr)
        tumor_type = cls["tumor_type"]
        if tumor_type == "unknown":
            raise HTTPException(422, "Cannot auto-classify. Provide tumor_type explicitly.")

    tumor_type = tumor_type.lower()
    if tumor_type not in ["pituitary", "meningioma", "glioma"]:
        raise HTTPException(400, "tumor_type must be: pituitary | meningioma | glioma")

    try:
        if tumor_type == "pituitary":
            pred = run_pituitary_inference(arr)
        elif tumor_type == "meningioma":
            pred = run_meningioma_inference(arr)
        else:
            pred = run_glioma_inference(arr)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    pid     = file.filename.replace(".nii.gz","").replace(".nii","")
    img_b64 = create_visualization(arr, pred, tumor_type, pid)
    stats   = compute_stats(pred, tumor_type)

    return JSONResponse({
        "filename":    file.filename,
        "tumor_type":  tumor_type,
        "input_shape": list(arr.shape),
        "pred_shape":  list(pred.shape),
        "labels":      TUMOR_LABELS[tumor_type],
        "stats":       stats,
        "visualization_base64": img_b64,
    })


@app.post("/pipeline")
async def pipeline(file: UploadFile = File(...)):
    """
    Full pipeline:
    1. Auto-classify tumor type
    2. Route to correct model
    3. Return segmentation + visualization
    """
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "Only .nii/.nii.gz supported")

    data = await file.read()
    arr, _ = load_nifti_from_bytes(data)

    # Step 1 — Classify
    classification = classify_tumor(arr)
    tumor_type     = classification["tumor_type"]
    if tumor_type == "unknown":
        raise HTTPException(422,
            f"Cannot determine tumor type from shape {arr.shape}. "
            f"Use /segment with explicit tumor_type.")

    # Step 2 — Segment
    try:
        if tumor_type == "pituitary":
            pred = run_pituitary_inference(arr)
        elif tumor_type == "meningioma":
            pred = run_meningioma_inference(arr)
        else:
            pred = run_glioma_inference(arr)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    # Step 3 — Visualize + stats
    pid     = file.filename.replace(".nii.gz","").replace(".nii","")
    img_b64 = create_visualization(arr, pred, tumor_type, pid)
    stats   = compute_stats(pred, tumor_type)

    return JSONResponse({
        "pipeline_steps":   ["classify", "segment", "visualize"],
        "filename":         file.filename,
        "classification":   classification,
        "segmentation": {
            "tumor_type":  tumor_type,
            "input_shape": list(arr.shape),
            "pred_shape":  list(pred.shape),
            "labels":      TUMOR_LABELS[tumor_type],
            "stats":       stats,
        },
        "visualization_base64": img_b64,
    })


@app.on_event("startup")
async def startup_event():
    print("🚀 Preloading models...")
    for fold in [0, 1]:
        try: get_pituitary_model(fold)
        except Exception as e: print(f"⚠️  Pituitary fold {fold}: {e}")
    try: get_meningioma_model()
    except Exception as e: print(f"⚠️  Meningioma: {e}")
    try: get_glioma_model()
    except Exception as e: print(f"⚠️  Glioma: {e}")
    print(f"✅ Ready. Models loaded: {list(_models.keys())}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
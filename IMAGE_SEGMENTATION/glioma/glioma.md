## 1. Clinical Context & BraTS Nested Sub-Region Hierarchy

Diffuse gliomas (Glioblastoma / HGG and Lower-Grade Glioma / LGG) are infiltrative primary brain neoplasms. Because tumor cells invade surrounding healthy brain tissue without a clear capsule, the **MICCAI BraTS** challenge defines three hierarchical, overlapping compartments:

$$\begin{aligned}
\mathbf{WT} &\text{ (Whole Tumor)} = \text{Label 1 (Necrosis)} \cup \text{Label 2 (Edema)} \cup \text{Label 4 (Enhancing Rim)} \\
\mathbf{TC} &\text{ (Tumor Core)} = \text{Label 1 (Necrosis)} \cup \text{Label 4 (Enhancing Rim)} \\
\mathbf{ET} &\text{ (Enhancing Tumor)} = \text{Label 4 (Actively Angiogenic Proliferating Edge)}
\end{aligned}$$

---

## 2. Multi-Parametric MRI Sequence Synergy

The model requires 4 co-registered MRI sequences:
* **T1-Native**: Baseline brain anatomy and surgical landmarks.
* **T1-CE**: Blood-brain barrier breakdown; defines the outer border of the **Enhancing Tumor (ET)**.
* **T2**: Fluid contrast; highlights the solid non-enhancing tumor core and ventricles.
* **T2-FLAIR**: Attenuates free CSF, making parenchymal vasogenic edema bright; defines the **Whole Tumor (WT)** boundary.

---

## 3. Network Architecture & Forward Pass Formulation

The architecture is a 3D Deeply Supervised Residual Network (**BrainTumNet / SegResNet**, 18.8M parameters):
* **Input**: 4 Channels `[Batch, 4, 128, 128, 128]`.
* **Multi-Label Sigmoid Activation**: Unlike mutually exclusive softmax classification, the output head uses **three independent Sigmoid units**:
  $$\hat{\mathbf{Y}} = \sigma(\mathbf{Z}) \in [0, 1]^{3 \times 128 \times 128 \times 128}$$
  where Channel 0 predicts $\mathbf{TC}$, Channel 1 predicts $\mathbf{WT}$, and Channel 2 predicts $\mathbf{ET}$. This permits the network to naturally learn the anatomical containment property ($\mathbf{ET} \subseteq \mathbf{TC} \subseteq \mathbf{WT}$).

---

## 4. Loss Formulation & Gradient Optimization

$$\mathcal{L}_{\text{BraTS}} = 2.0 \cdot \mathcal{L}_{\text{Tversky}}(\hat{\mathbf{Y}}_{\text{TC}}, \mathbf{Y}_{\text{TC}}) + 1.0 \cdot \mathcal{L}_{\text{DiceCE}}(\hat{\mathbf{Y}}_{\text{WT}}, \mathbf{Y}_{\text{WT}}) + 1.0 \cdot \mathcal{L}_{\text{DiceCE}}(\hat{\mathbf{Y}}_{\text{ET}}, \mathbf{Y}_{\text{ET}})$$

### Why Tumor Core (TC) Has 2.0x Weight:
The surgical objective in glioma resection is maximum safe resection of the **Tumor Core**. Edema (WT) is treated medically with dexamethasone, while enhancing tumor (ET) is easily visible. The non-enhancing tumor core is the most difficult to delineate on MRI. Weighting TC at $2.0\times$ forces larger backpropagation gradients to flow into the intermediate residual units, preventing under-segmentation.

---

## 5. Benchmark Performance Metrics

| Glioma Sub-Region | Voxel Composition | Mean Dice Score | Peak Dice |
| :--- | :--- | :---: | :---: |
| **Whole Tumor (WT)** | Labels 1 + 2 + 4 | **0.8971** | **0.9520** |
| **Tumor Core (TC)** | Labels 1 + 4 | **0.8420** | **0.9150** |
| **Enhancing Tumor (ET)** | Label 4 | **0.8080** | **0.8840** |
| **Composite Mean** | $(\text{WT} + \text{TC} + \text{ET}) / 3$ | **0.8490** | **0.9170** |

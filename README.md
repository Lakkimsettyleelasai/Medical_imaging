# Comprehensive 3D Brain Disease Diagnostics, Multi-Modal Imaging, and Volumetric Segmentation

A multi-task deep learning platform for automated neuro-radiological diagnosis, lesion classification, 3D volumetric segmentation, and surgical planning across multi-parametric MRI, CT, and multi-modal neuroimaging scans.

---

## Overview

This platform is designed as an end-to-end clinical neuro-radiology system capable of analyzing both multi-sequence MRI and CT scans. The architecture combines disease triage classification, high-resolution 3D volumetric segmentation, and surgical risk assessment into a single modular framework.

The primary foundation began with neuro-oncology (Pituitary Macroadenoma with internal carotid artery tracking, Meningioma, and Diffuse Glioma). The system is now expanding into a universal neuroimaging platform that encompasses acute cerebrovascular events, neurodegenerative conditions, neuro-inflammatory diseases, and traumatic brain injury.

---

## Comprehensive Neuro-Imaging Pathology Scope

The platform is architected to handle the full spectrum of brain diseases diagnosable through MRI and CT:

### 1. Neuro-Oncology (Current SOTA Baseline)
* **Diffuse Gliomas (HGG and LGG)**: Multi-compartment segmentation of Whole Tumor, Tumor Core, and Enhancing Tumor across T1, T1-CE, T2, and FLAIR.
* **Meningiomas**: Dural tail infiltration, peritumoral vasogenic edema, and solid tumor mass boundaries.
* **Pituitary Adenomas and Sellar Pathologies**: Sella-centric 0.5 mm micro-anatomy with bilateral Internal Carotid Artery (ICA) encasement analysis for Knosp surgical classification.
* **Cerebral Metastases**: Multi-focal contrast-enhancing lesion detection and volume tracking.
* **Vestibular Schwannomas (Acoustic Neuromas)**: Cerebellopontine angle localization and cranial nerve margin analysis.

### 2. Cerebrovascular and Acute Neuro-Emergencies
* **Acute Ischemic Stroke**: Diffusion-weighted imaging (DWI / ADC) core infarct segmentation paired with CT Perfusion (CTP) penumbra mismatch calculation.
* **Intracranial Hemorrhage (ICH)**: Non-Contrast Head CT (NCCT) automated detection and volume quantification of epidural, subdural, subarachnoid (SAH), and intraparenchymal hemorrhages.
* **Vascular Anomalies**: Automated identification of cerebral aneurysms and arteriovenous malformations (AVM) on CT Angiography (CTA) and MR Angiography (MRA).

### 3. Neurodegenerative and Cognitive Disorders
* **Alzheimer's Disease and Dementia**: Automated volumetric quantification of hippocampal atrophy, ventricular enlargement, and cortical thinning on 3D T1 MPRAGE scans.
* **Vascular Dementia**: White matter hyperintensity (WMH) Fazekas scale burden scoring on T2-FLAIR.

### 4. Demyelinating and Inflammatory Diseases
* **Multiple Sclerosis (MS)**: Automated longitudinal tracking of periventricular, juxtacortical, and infratentorial plaque burden following McDonald diagnostic criteria.
* **Intracranial Infections**: Detection of brain abscesses, empyemas, and encephalitis patterns.

### 5. Traumatic Brain Injury (TBI) and Skull Base Trauma
* **Bone and Soft Tissue Trauma**: Bone-window CT segmentation for skull fractures paired with SWI / GRE detection of punctate microhemorrhages in Diffuse Axonal Injury (DAI).

### 6. Healthy / Normal Brain Baseline
* Quality assurance, anatomical normal-variant identification, and age-matched normative volume indexing to eliminate false-positive surgical recommendations.

---

## Supported Imaging Modalities

* **Magnetic Resonance Imaging (MRI)**:
  * Structural: T1-Weighted, T1-Contrast Enhanced (T1-CE), T2-Weighted, T2-FLAIR.
  * Physiological & Functional: Diffusion-Weighted Imaging (DWI), Apparent Diffusion Coefficient (ADC), Susceptibility-Weighted Imaging (SWI / GRE), Perfusion-Weighted Imaging (PWI), and MR Angiography (MRA).
* **Computed Tomography (CT)**:
  * Non-Contrast Head CT (NCCT), CT Angiography (CTA), and CT Perfusion (CTP).
* **Positron Emission Tomography (PET)**:
  * Co-registered FDG-PET and Amyloid-PET for metabolic and molecular neuro-analysis.

---

## Benchmark Scores Achieved

### 1. Pituitary Macroadenoma and Bilateral Internal Carotid Artery (ICA)
* **Dataset**: Pandit et al. (Nature Scientific Data, 2025), 99 complete patient volumes at native 0.5 mm isotropic resolution.
* **Modalities**: T1-CE and co-registered T2.
* **Method**: 2-Stage Sella-Centric Zoom Cascade (1.0 mm coarse localization followed by 0.5 mm fine SegResNet).

| Evaluation Metric | Grand Cohort Mean (99 Cases) | Unseen Validation Mean (19 Cases) | Peak Score (Unseen) |
| :--- | :---: | :---: | :---: |
| **Pituitary Tumor Dice** | **0.8669** | **0.8191** | **0.9238** |
| **Bilateral Carotids (ICA) Dice** | **0.7376** | **0.6699** | **0.8364** |
| **Overall Mean Dice** | **0.8023** | **0.7445** | **0.8677** |

*Showcase Case (Patient 96)*: Tumor Dice: **0.8990** | Carotid Dice: **0.8364** | Composite: **0.8677**.

---

### 2. Meningioma Multi-Compartment Segmentation
* **Dataset**: MICCAI BraTS-MEN Challenge, evaluated across 300 independent unseen test patients.
* **Modalities**: T1-CE and T2-FLAIR.
* **Method**: 3D SegResNet with deep supervision and Focal-Tversky loss.

| Sub-Region | Class Description | Validation Dice Score |
| :--- | :--- | :---: |
| **Necrotic Core** | Class 1 (Internal ischemic degeneration) | **0.9078** |
| **Peritumoral Edema** | Class 2 (Vasogenic fluid in white matter) | **0.8837** |
| **Enhancing Tumor** | Class 3 (Active contrast-enhancing mass and dural tail) | **0.8080** |
| **True Cohort Mean** | Average across all 300 unseen cases | **0.8665** |
| **Whole Evaluation Mean** | Aggregate volume evaluation | **0.8971** |

*Peak Case (BraTS-MEN-00010-000)*: Mean Dice: **0.9836** (Necrotic: 1.000, Edema: 1.000, Enhancing: 0.951).

---

### 3. Diffuse Glioma Multi-Compartment Segmentation
* **Dataset**: MICCAI BraTS Cohort (HGG and LGG).
* **Modalities**: T1, T1-CE, T2, and T2-FLAIR.
* **Method**: 3D BrainTumNet / SegResNet with hierarchical multi-label sigmoid outputs and 2.0x Tumor Core weighted backpropagation.

| Compartment | Anatomical Inclusion | Mean Dice Score | Peak Dice Score |
| :--- | :--- | :---: | :---: |
| **Whole Tumor (WT)** | Edema + Core + Enhancing Rim | **0.8971** | **0.9520** |
| **Tumor Core (TC)** | Gross resectable mass | **0.8420** | **0.9150** |
| **Enhancing Tumor (ET)** | Actively proliferating angiogenic rim | **0.8080** | **0.8840** |
| **Composite Mean** | Average across all sub-regions | **0.8490** | **0.9170** |

---

## What Work Has Been Done

1. **Native 0.5 mm Sella Cascade Implementation**:
   * Proved that standard 1.0 mm scans physically cap carotid artery Dice scores at ~0.65 due to partial volume averaging across 3-voxel vessel widths.
   * Engineered a 2-stage cascade: Stage 1 scans the whole brain at 1.0 mm to locate the sella turcica centroid (zc, yc, xc), and Stage 2 crops a native 0.5 mm 128x128x128 cube to dedicate all convolutional capacity exclusively to the adenoma and internal carotid arteries.
   * Evaluated all 99 cases and generated 3-plane anatomical visualizations for every patient.

2. **Meningioma and Glioma Delineation**:
   * Validated 3D deep models on standardized BraTS multi-center cohorts.
   * Integrated class-imbalance-aware losses (Focal-Tversky and Asymmetric DiceCE) to capture small necrotic cores and fine enhancing margins without background noise.

3. **Workspace Cleanup and Code Organization**:
   * Retired legacy 2D slice-based models and outdated nnU-Net caches.
   * Structured modules into clean `IMAGE_CLASSIFICATION/` and `IMAGE_SEGMENTATION/` packages.

---

## What Needs To Be Done

1. **Universal Multi-Disease Classifier Training**:
   * Retrain and validate the multi-class disease classifier to integrate the new 0.5 mm pituitary dataset alongside Glioma, Meningioma, Acute Stroke/Hemorrhage, and Normal controls.
   * Establish automated modality detection and sequence identification to route incoming DICOM/NIfTI scans to the appropriate diagnostic pathway.

2. **Multi-Modal Expansion to CT and Acute Pathologies**:
   * Integrate Non-Contrast CT (NCCT) models for rapid triage of acute intracranial hemorrhages (ICH) and midline shifts within emergency department workflows.
   * Implement stroke infarct core and penumbra volumetric mismatch pipelines for DWI and CT Perfusion.

3. **Rebuilding the Unified Full-Stack Platform**:
   * **Backend**: Build a modern, asynchronous FastAPI backend capable of ingesting multi-modal DICOM/NIfTI studies, queueing GPU inference jobs, and serving 3D mesh representations.
   * **Frontend**: Build a responsive interface featuring multi-planar (Axial, Coronal, Sagittal) synchronized slice viewers, real-time 3D interactive mesh rendering (Three.js / GLB), and automated Knosp surgical risk grading.
   * **Clinical Reporting Engine**: Integrate a Vision-Language Model (VLM) to generate structured radiologist draft reports with automated volumetric measurements and comparative prior-scan analytics.

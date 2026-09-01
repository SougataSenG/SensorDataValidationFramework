# Sensor Validation Framework (SVF) — GTRE Data Implementation

A deep learning pipeline for sensor fault **Detection, Isolation, and Accommodation (DIA)** on the GTRE Real engine dataset. The framework implements a 1D-CNN Convolutional Autoencoder for anomaly detection, three parallel fault classifiers (Multi-Head ANN, BiLSTM, Transformer Encoder), GradientSHAP explainability, Monte Carlo Dropout uncertainty quantification, and virtual sensor accommodation — realising the classical NASA ADIA analytical redundancy paradigm through entirely data-driven components.

---

## Repository Structure

```
Notebooks/
|   ├── NB1_LabDataLoader_final.ipynb   	# Data loading, cruise extraction, fault injection
|   ├── NB2_LabAutoencoder.ipynb       		# 1D-CNN AE training and threshold calibration
|   ├── NB3_LabClassification.ipynb     	# ANN / BiLSTM / Transformer training and evaluation
|   ├── NB4_SHAP.ipynb                		# GradientSHAP attribution analysis
|   ├── NB5_Diagnosis_VirtualSensors.ipynb      # Diagnostic orchestrator and virtual sensor accommodation
|   ├── NB6_LabTesting_Final.ipynb		# Controlled Testings and Real Lab Engine Failure Test
│
├── data/
│   └── Lab/                  	  # Place raw .csv lab dataset files here (created manually)
│
├── processed_data/               # Auto-created: scaled windows, fault-injected dataset
├── checkpoints/                  # Auto-created: saved model weights (.pt)
├── plots/                        # Auto-created: all figures and training curves
|
├── lab_data_utils.py                 # Script for loading the dataset and extracting the cruise phase, data preprocessing
├── explainability.py             # Script for implementing GradientSHAP
├── fault_engine.py               # used for injecting faults into the dataset
├── feature_engineering.py        # Helps to extract multiple features(statistical, fft...) of a window of dataset
└── models.py                     # Design and implementation of all the models used in the work
```

> `processed_data/`, `checkpoints/`, and `plots/` are created automatically when the first cell of NB1 is executed. Only the `data/ncmapss/` directory needs to be created manually.

---

## Setup

### Requirements

```bash
pip install torch numpy pandas h5py scikit-learn matplotlib captum ruptures
```


### Data Preparation

1. Create the data directory:
   ```bash
   mkdir -p data/lab
   ```

2. Copy the GTRE dataset and keep it in 'data/lab/'.
   (If you are using the GTRE Machine, the files are already saved in the DataSet folder)

   The directory should look like:
   ```
   data/lab/
   └── 1590_Binary_Data_Dump_1.csv
   └──1591_Binary_Data_Dump_1.csv
   └──1595_Binary_Data_Dump_1.csv
   └──1596_Binary_Data_Dump_1.csv
   └──1600_Binary_Data_Dump_1.csv
   .
   .
   └──1681_Binary_Data_Dump_1.csv
   ```

---

## Running the Pipeline

Run the five notebooks **sequentially**. Each notebook saves its outputs to disk so downstream notebooks can load them without recomputation.

| Notebook | Purpose | Key Outputs |
|---|---|---|
| `NB1_DataLoader` | HDF5 loading, cruise-phase extraction, MinMax scaling, fault injection, balanced dataset assembly | `processed_data/windows_*.npy`, fitted scaler |
| `NB2_Autoencoder` | 1D-CNN AE training on nominal cruise windows, anomaly threshold calibration at 95th percentile | `checkpoints/autoencoder.pt`, threshold τ |
| `NB3_Classifiers` | AE residual augmentation, ANN / BiLSTM / Transformer training, stratified evaluation | `checkpoints/[model].pt`, per-class metrics |
| `NB4_SHAP` | GradientSHAP attribution per fault class, sensor importance ranking and visualisation | `plots/shap_*.png` |
| `NB5_Diagnosis` | End-to-end diagnostic orchestrator, structured 5-case test evaluation, virtual sensor accommodation | `plots/diagnosis_*.png` |

> **Resuming a notebook:** Every notebook contains a dedicated **reload cell**. If you have already executed a notebook once, run only the reload cell to restore all saved artefacts from disk — no need to rerun training from scratch.

---

## Framework Overview

```
Raw Sensor Window  (T × C)
        │
        ▼
┌──────────────────────┐
│  Stage 1             │
│  AE Anomaly Scoring  │  →  Anomaly flag  +  per-channel reconstruction error
└──────────┬───────────┘
           │  (if anomaly detected)
           ▼
┌──────────────────────┐
│  Stage 2             │  →  Fault type (8-class)
│  BiLSTM Classifier   │  →  Faulty sensor ID (14-class)
│  + MC-Dropout        │  →  Fault magnitude (regression)
│                      │  →  Epistemic confidence estimate
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Stage 3             │
│  GradientSHAP        │  →  Per-sensor, per-timestep attribution map
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Stage 4             │
│  Virtual Sensor      │  →  Faulty channel replaced by AE reconstruction
│  Accommodation       │
└──────────────────────┘
```

---

## Fault Taxonomy

The training dataset is constructed from physics-informed synthetic fault injection across **8 balanced classes** (241,008 windows total, 30,126 per class), following the taxonomy of Balaban et al. (2009):

| Class | Description | Source |
|---|---|---|
| Nominal | No fault present | Real nominal cruise data |
| Bias | Constant additive offset from onset timestep | Synthetic — Case 1 |
| Scaling | Multiplicative gain error from onset timestep | Synthetic — Case 1 |
| Drift | Linearly growing deviation to window end | Synthetic — Case 1 |
| Intermittent | Random spike excursions (p = 0.25 per timestep) | Synthetic — Case 1 |
| Compound | Two sequential fault modes on one channel | Synthetic — Case 2 |
| MultiSensor | Simultaneous independent faults on two channels | Synthetic — Case 3 |
| SystemFault | Real engine failure data (1 csv file provided) | Provided by the lab |

---

## Key Results

| Metric | Value |
|---|---|
| Anomaly detection recall (17 structured scenarios) | **100%** |
| BiLSTM overall accuracy — N-CMAPSS (8-class) | **75%** |
| BiLSTM macro F1 — N-CMAPSS | **0.72** |
| BiLSTM overall accuracy — Laboratory data (7-class) | **81.4%** |
| BiLSTM macro F1 — Laboratory data | **0.813** |
| Random-chance baseline — N-CMAPSS | 12.5% |
| Bias / Scaling / Drift F1 (N-CMAPSS, BiLSTM) | 0.86 / 0.81 / 0.82 |
| Real engine failure detection AUC | **1.000** |

---

## Architecture Summary

| Component | Architecture | Parameters |
|---|---|---|
| Anomaly Detector | 1D-CNN Autoencoder (3 enc. blocks, bottleneck d=64, 3 dec. blocks) | ~2.1 M |
| Multi-Head ANN | FC(196→256→128→64) + 3 output heads | 96,023 |
| Multi-Head BiLSTM | 2-layer BiLSTM (d=128/dir) + 3 output heads | 678,871 |
| Multi-Head Transformer | 2 encoder layers, H=4 heads, d_model=64 + 3 output heads | 105,879 |

All classifiers share the same combined loss:

```
L = 1.0 × L_CE(fault type) + 0.5 × L_CE(sensor ID) + 0.3 × L_MSE(magnitude)
```

---

### Suggested Future Directions

**Transient phase extension**
The current framework is restricted to steady-state cruise-phase windows. Potential directions include:
- Phase-conditional autoencoders and classifiers trained separately for climb, descent, and take-off phases
- Physics-informed normalisation using auxiliary variables (altitude, Mach number, TRA) to remove flight-phase-driven variance while preserving fault signatures
- A phase-detection pre-stage to route incoming windows to the appropriate phase-specific model

**Multi-label fault classification**
Compound and MultiSensor faults are currently forced into single-label predictions. A multi-label output head or a hierarchical classifier (first predict fault count, then predict each fault type and channel) would improve recall on these classes significantly.

**Sensor-specific adaptive thresholding**
Replace the global 95th-percentile anomaly threshold with per-channel, flight-condition-aware thresholds — particularly relevant for sensor P2, which is partially driven by external atmospheric conditions and currently shows systematic low-confidence misclassification.

**Online / edge deployment**
The Transformer's ~106K parameter count makes it a strong candidate for INT8 quantisation and ONNX export for deployment on low-power embedded hardware compatible with FADEC-adjacent monitoring systems.

---

## Acknowledgements

This work is conducted at the **Gas Turbine Research Establishment (GTRE), DRDO, Bangalore** under the supervision of:


- **Dr. Yogeshwar Singh Dadwhal**, Assistant Professor, Department of Applied Mathematics, DIAT, Pune (Supervisor)
- **Vikas Rama Rao**, Scientist 'F', AI & ML Division, GTRE (Co-Supervisor)
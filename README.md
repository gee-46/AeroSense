 
<div align="center">

# AeroSense

### Deep Learning-Based Acoustic Anomaly Detection for UAV Propulsion Systems

**From acoustic recordings to explainable UAV propeller fault detection.**

<br />

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Frontend-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tests](https://img.shields.io/badge/Tests-38%20passing-2ea44f?style=for-the-badge)]()

</div>

---

## Overview

**AeroSense** is a full-stack deep learning system for detecting UAV
propeller mechanical damage from acoustic recordings.

The system combines two complementary trained PyTorch models:

- **CNN classifier** — identifies known propeller-damage classes.
- **Convolutional autoencoder** — detects acoustic anomalies through
  reconstruction error.
- **Decision engine** — combines model outputs into
  `NORMAL`, `ANOMALY`, or `UNKNOWN_ANOMALY`.
- **Grad-CAM** — provides frequency/time attribution from the classifier's
  actual gradients.
- **FastAPI backend** — exposes the complete inference and analytics pipeline.
- **React + TypeScript frontend** — provides analysis, diagnostics, history,
  analytics, and model monitoring.

> **All inference results are produced by real trained models.**
> The application does not use simulated predictions or placeholder model
> outputs.

---

## Results at a Glance

| Component | Result |
|---|---:|
| CNN Test Accuracy | **96.0%** |
| CNN Macro F1 | **0.961** |
| Autoencoder Test F1 | **0.928** |
| Autoencoder ROC-AUC | **0.909** |
| CNN Parameters | **106,148** |
| Autoencoder Parameters | **51,041** |
| Held-out Test Segments | **3,114** |
| Automated Tests | **75 passing, 6 skipped** (skips need the raw dataset; see Testing) |

---

## Why AeroSense?

A conventional classifier can force an abnormal acoustic signal into one of
its known classes even when the signal does not confidently match anything
seen during training.

AeroSense uses two complementary models:

```text
                         Acoustic Recording
                                │
                                ▼
                         Audio Preprocessing
                                │
                                ▼
                       Log-Mel Spectrogram
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              CNN Classifier          Autoencoder
                    │                       │
             Fault + Confidence       Reconstruction
                                         Error
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         Decision Engine
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
              NORMAL         ANOMALY        UNKNOWN
                                             ANOMALY
                                │
                                ▼
                             Grad-CAM
                                │
                                ▼
                         Diagnostic Report
````

This allows the system to distinguish between a **known fault** and an
**acoustically abnormal signal that does not confidently match a known
fault class**.

---

# Application

The trained models are exposed through a FastAPI backend and consumed by a
React + TypeScript frontend.

## Dashboard

The dashboard provides:

* Live model status
* Detection statistics
* Anomaly trends
* Fault distribution
* Recent predictions

## Acoustic Analysis

The analysis workflow provides:

* Drag-and-drop WAV upload
* Audio metadata
* Real waveform visualization
* Context fields
* Live inference state
* CNN classification
* Classification confidence
* Autoencoder anomaly score
* Severity assessment
* Grad-CAM explanation

## Prediction History

* Searchable prediction history
* Status filtering
* Device filtering
* Pagination
* Prediction details

## Diagnostic Report

Each prediction can be inspected through an engineering-style report containing:

* Signal information
* Predicted fault
* Confidence
* Anomaly score
* Severity
* Model attribution
* Frequency range
* Time range
* Device/environment metadata

## Analytics

The analytics dashboard exposes real backend aggregate data including:

* Total predictions
* Normal/anomalous distribution
* Fault distribution
* Anomaly trends

## Model Monitoring

The Models page displays metadata for the models that are actually trained
and active.

Untrained CRNN/Transformer architectures are **not** presented as deployed
models.

---

# System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    React + TypeScript                        │
│                                                              │
│  Dashboard │ Analyze │ History │ Analytics │ Models │ Detail │
└──────────────────────────────┬───────────────────────────────┘
                               │
                         REST / JSON
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                         FastAPI                              │
│                                                              │
│  Validation │ Preprocessing │ Inference │ Analytics │ DB    │
└──────────────────────┬───────────────────────┬───────────────┘
                       │                       │
                       ▼                       ▼
              ┌────────────────┐      ┌──────────────────┐
              │  CNN Classifier │      │   Autoencoder   │
              │                │      │                  │
              │ Known faults   │      │ Anomaly score    │
              └───────┬────────┘      └────────┬─────────┘
                      │                        │
                      └───────────┬────────────┘
                                  ▼
                         ┌────────────────┐
                         │ Decision Engine│
                         └───────┬────────┘
                                 │
                       ┌─────────┼─────────┐
                       ▼         ▼         ▼
                    NORMAL   ANOMALY   UNKNOWN
                                         ANOMALY
                                 │
                                 ▼
                         ┌──────────────┐
                         │   Grad-CAM   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   SQLite DB  │
                         └──────────────┘
```

---

# Machine Learning Pipeline

```text
Raw WAV
   │
   ▼
16 kHz Mono
   │
   ▼
2.0s Windows / 1.0s Hop
   │
   ▼
Peak Normalization
   │
   ▼
64-band Log-Mel Spectrogram
   │
   ▼
Train-Split Standardization
   │
   ▼
(1, 64, 126) Tensor
   │
   ├──────────────────────┐
   ▼                      ▼
CNN Classifier       Autoencoder
   │                      │
   ▼                      ▼
Fault Class          Reconstruction
Confidence              Error
   │                      │
   └───────────┬──────────┘
               ▼
        Decision Engine
               │
               ▼
       Grad-CAM + Report
```

Preprocessing is implemented once in:

```text
common/audio_features.py
common/config.py
```

The same implementation is used by training and inference to prevent
train/inference preprocessing skew.

---

# Dataset

**Source:**
[TII UAV Propeller Anomaly Audio Dataset](https://github.com/tiiuae/UAV-Propeller-Anomaly-Audio-Dataset)

Technology Innovation Institute, ICASSP 2022.

| Property                | Value                                          |
| ----------------------- | ---------------------------------------------- |
| Classes                 | `normal`, `1_broken`, `2_broken`, `3_4_broken` |
| Recordings              | 293                                            |
| Normal recordings       | 78                                             |
| `1_broken` recordings   | 72                                             |
| `2_broken` recordings   | 71                                             |
| `3_4_broken` recordings | 72                                             |
| Sample rate             | 16 kHz                                         |
| Channels                | Mono                                           |
| Format                  | 16-bit PCM WAV                                 |
| Total audio             | ~5.7 hours                                     |
| Generated segments      | 18,569                                         |

The dataset represents **propeller damage severity**.

It does not contain bearing faults, motor faults, or fixed-wing aircraft
engine faults.

---

# Leakage-Safe Dataset Split

AeroSense performs the split at the **recording level**, rather than
randomly splitting individual segments.

This prevents overlapping or adjacent segments from the same flight
recording from appearing in multiple splits.

| Split      |   Segments |
| ---------- | ---------: |
| Train      |     12,835 |
| Validation |      2,620 |
| Test       |      3,114 |
| **Total**  | **18,569** |

Standardization statistics are calculated using the training split only and
reused during inference.

Stored in:

```text
models/feature_stats.json
```

---

# Preprocessing

Each recording follows the same preprocessing pipeline:

```text
WAV
 │
 ├── Resample → 16 kHz
 ├── Convert → Mono
 ├── Segment → 2.0 s windows / 1.0 s hop
 ├── Peak normalize
 ├── Log-Mel Spectrogram
 │      ├── n_mels = 64
 │      ├── n_fft  = 1024
 │      ├── hop    = 256
 │      ├── fmin   = 20 Hz
 │      └── fmax   = 8000 Hz
 │
 └── Train-split standardization
             │
             ▼
        (1, 64, 126)
```

---

# Data Augmentation

Applied only to training data:

* Gaussian noise
* Random amplitude gain
* Circular time shifting
* Frequency masking
* Time masking

This prevents augmented samples from contaminating validation or test
evaluation.

---

# Model Architectures

## CNN Classifier

```text
Conv2D
   ↓
BatchNorm
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2D
   ↓
BatchNorm
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2D
   ↓
BatchNorm
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2D
   ↓
BatchNorm
   ↓
ReLU
   ↓
MaxPool
   ↓
Global Average Pool
   ↓
FC 128 → 64
   ↓
FC 64 → 4
```

**Parameters:** 106,148

Classes:

```text
normal
1_broken
2_broken
3_4_broken
```

The model is intentionally compact for the available CPU-only training
environment and the relatively small number of source recordings.

---

## Convolutional Autoencoder

```text
Input Spectrogram
       │
       ▼
Strided Conv Encoder
       │
       ▼
32-D Latent Space
       │
       ▼
Transposed Conv Decoder
       │
       ▼
Reconstructed Spectrogram
```

**Parameters:** 51,041

The autoencoder is trained **only on normal acoustic samples**.

At inference:

```text
Anomaly Score = Reconstruction MSE
```

A high reconstruction error indicates that the acoustic pattern differs
from the normal training distribution.

---

# Training Methodology

Both training pipelines use:

* Adam optimizer
* `ReduceLROnPlateau`
* Early stopping
* Best-checkpoint selection
* Fixed random seed (`42`)
* Class-balanced cross-entropy for the CNN
* Checkpointed/resumable training

Training scripts:

```text
training/
├── dataset.py
├── prepare_dataset.py
├── train_cnn.py
└── train_autoencoder.py
```

Training metadata is stored in:

```text
models/
├── cnn_training_log.json
├── cnn_trainer_state.json
├── autoencoder_training_log.json
└── autoencoder_trainer_state.json
```

---

# Results

All reported metrics are from the **held-out test split**.

## CNN Classifier

| Metric                |              Result |
| --------------------- | ------------------: |
| Accuracy              |           **96.0%** |
| Macro F1              |           **0.961** |
| Validation Macro F1   |           **0.947** |
| Parameters            |         **106,148** |
| Average CPU inference | **~2.6 ms/segment** |

### Confusion Matrix

```text
                 predicted
              normal 1_brk 2_brk 3_4_brk

true normal     808     0     1      9
     1_broken     0   658    14      5
     2_broken     3     7   762     31
     3_4_broken   0     1    53    762
```

The primary confusion occurs between `2_broken` and `3_4_broken`, the two
adjacent damage-severity classes.

Evaluation artifact:

```text
evaluation/results/cnn_confusion_matrix.png
```

---

## Autoencoder Anomaly Detector

The anomaly threshold is **fitted using the validation set** and then
evaluated once on the held-out test set.

| Metric    |      Result |
| --------- | ----------: |
| Threshold | **0.02494** |
| Precision |   **0.920** |
| Recall    |   **0.936** |
| F1        |   **0.928** |
| ROC-AUC   |   **0.909** |

Evaluation artifact:

```text
evaluation/results/autoencoder_roc.png
```

---

# Unknown Anomaly Detection

The decision engine combines the outputs of both trained models.

### NORMAL

Reconstruction error is below the fitted anomaly threshold and the
classifier does not confidently identify a known fault.

### ANOMALY

Reconstruction error exceeds the threshold and the classifier confidently
identifies a known fault.

### UNKNOWN_ANOMALY

Reconstruction error exceeds the threshold, but the classifier is not
confident enough to assign a known fault class.

```text
Known fault
     │
     └── Classifier confident
              │
              ▼
           ANOMALY


Acoustically abnormal
     │
     └── Classifier uncertain
              │
              ▼
       UNKNOWN_ANOMALY
```

The system therefore avoids forcing every abnormal signal into one of the
known training classes.

---

# Explainability

AeroSense implements **real Grad-CAM** over the CNN's final convolutional
layer.

For each analyzed recording, the backend can return:

* Grad-CAM heatmap
* Mel-spectrogram overlay
* Important frequency range
* Important time range

The attribution is computed from the trained model's actual gradients on
the actual input recording.

It is not hardcoded or templated.

Implementation:

```text
backend/app/services/explainability_service.py
```

---

# Backend

The backend is built using:

* FastAPI
* Pydantic
* SQLAlchemy
* SQLite
* Uvicorn
* PyTorch

### Core responsibilities

```text
Audio Validation
      ↓
Preprocessing
      ↓
Model Loading
      ↓
CNN + Autoencoder Inference
      ↓
Decision Engine
      ↓
Grad-CAM
      ↓
Persistence
      ↓
Analytics
```

Models are loaded through a shared model loader and reused across requests.

---

# API

Base URL:

```text
http://localhost:8000/api/v1
```

| Method   | Endpoint                        | Description            |
| -------- | ------------------------------- | ---------------------- |
| `GET`    | `/health`                       | API health             |
| `GET`    | `/health/model`                 | Model availability     |
| `POST`   | `/audio/upload`                 | Validate/probe audio   |
| `POST`   | `/audio/analyze`                | Run complete inference |
| `GET`    | `/predictions`                  | List predictions       |
| `GET`    | `/predictions/{id}`             | Retrieve prediction    |
| `DELETE` | `/predictions/{id}`             | Delete prediction      |
| `GET`    | `/models`                       | Model metadata         |
| `GET`    | `/models/active`                | Active model versions  |
| `GET`    | `/analytics/summary`            | Aggregate statistics   |
| `GET`    | `/analytics/fault-distribution` | Fault distribution     |
| `GET`    | `/analytics/anomaly-trend`      | Anomaly trends         |

Swagger:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

# Frontend

The frontend is built with:

* React
* TypeScript
* Vite
* Recharts
* Web Audio API

### Frontend structure

```text
frontend/src/
├── api/
│   ├── client.ts
│   ├── endpoints.ts
│   └── types.ts
│
├── components/
│   ├── analyze/
│   ├── background/
│   ├── charts/
│   ├── gradcam/
│   ├── layout/
│   ├── ui/
│   └── waveform/
│
├── hooks/
├── pages/
│   ├── DashboardPage.tsx
│   ├── AnalyzePage.tsx
│   ├── HistoryPage.tsx
│   ├── PredictionDetailPage.tsx
│   ├── AnalyticsPage.tsx
│   └── ModelsPage.tsx
│
├── styles/
└── utils/
```

The frontend communicates with the real `/api/v1/*` backend routes and does
not contain simulated inference data.

---

# Project Structure

```text
AeroSense/
│
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes/
│       ├── core/
│       ├── db/
│       ├── ml/
│       │   └── models/
│       ├── schemas/
│       ├── services/
│       └── utils/
│
├── frontend/
│   └── src/
│       ├── api/
│       ├── components/
│       ├── hooks/
│       ├── pages/
│       ├── styles/
│       └── utils/
│
├── common/
│   ├── audio_features.py
│   └── config.py
│
├── training/
│   ├── dataset.py
│   ├── prepare_dataset.py
│   ├── train_cnn.py
│   └── train_autoencoder.py
│
├── evaluation/
│   ├── evaluate.py
│   └── results/
│
├── models/
│   ├── aerosense_cnn_v1.pt
│   ├── aerosense_autoencoder_v1.pt
│   ├── feature_stats.json
│   └── registry.json
│
├── data/
│   └── processed/
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

# Quick Start

## Backend

### 1. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

### 4. Start the API

From the project root:

```powershell
$env:PYTHONPATH=".;backend"
uvicorn backend.app.main:app --reload --port 8000
```

Or:

```powershell
cd backend
$env:PYTHONPATH=".;.."
uvicorn app.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

---

## Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# Reproducing Training

The repository contains the complete training pipeline.

## 1. Obtain the dataset

```bash
git clone https://github.com/tiiuae/UAV-Propeller-Anomaly-Audio-Dataset.git
```

Place the dataset under:

```text
data/raw/
```

## 2. Prepare the dataset

```bash
python -m training.prepare_dataset
```

## 3. Train the CNN

```bash
python -m training.train_cnn \
  --epochs 12 \
  --batch-size 32 \
  --patience 3 \
  --resume \
  --time-budget-seconds 1200
```

The training process supports resumable checkpoints.

## 4. Train the autoencoder

```bash
python -m training.train_autoencoder \
  --epochs 15 \
  --batch-size 32 \
  --patience 4 \
  --resume \
  --time-budget-seconds 1200
```

## 5. Evaluate

```bash
python -m evaluation.evaluate
```

Results are written to:

```text
evaluation/results/
```

---

# Testing

```bash
python -m pytest tests/ -v -rs
```

Last verified run (Python 3.14.7, pinned `requirements.txt`): **75 passed, 6 skipped, 0 failed.**

The 6 skipped tests (`tests/test_inference_pipeline.py`) need real recordings from
`data/raw/` (not shipped; see *Reproducing Training*). They are **not** counted as
passing. Their coverage is replaced, not claimed, by `tests/test_api_integration.py`,
which pushes small *synthetic* audio through the **real checkpoints** and verifies
mechanics only: finite, normalized outputs, label order, threshold taken from
`registry.json`, persistence, Grad-CAM serving, cleanup, error envelopes. Synthetic
audio says nothing about accuracy; accuracy comes only from `evaluation/evaluate.py`
on the held-out split.

Coverage: audio validation, preprocessing, model architectures, checkpoint loading
safety (`test_model_loader.py`), decision-engine boundaries, health/model endpoints,
upload cleanup, Grad-CAM generation + HTTP retrieval + deletion, listing/filtering/
pagination, analytics aggregation, database persistence across app restart and
schema migration, OpenAPI contract.

Frontend:

```bash
cd frontend && npm ci && npx tsc -b && npm run lint && npm run build
```

The frontend has no automated UI tests; its API usage is checked against the backend
through the OpenAPI test above and by manual end-to-end runs.

---

# Clip-level aggregation

An upload is cut into 2 s segments (1 s hop). Each segment is scored by both models, then:

* class probabilities = **mean** of per-segment softmax vectors; the predicted class is its argmax;
* anomaly score = **mean** of per-segment reconstruction errors, compared with the threshold (`>=`).

Limitations: the threshold was fitted on *per-segment* validation errors, so applying it
to a clip mean is an approximation, and a short abnormal burst inside a long recording is
diluted by averaging. A confident `normal` prediction overrides a score within 10% above
the threshold (documented in `anomaly_service.py`). Clips shorter than 1 s are rejected;
clips longer than `MAX_AUDIO_DURATION_SECONDS` (default 600) are rejected.

# Data retention

Uploaded audio is **deleted after every request** (success or failure) and never served
back. Stored per prediction: derived results, audio metadata and the Grad-CAM overlay PNG
(`backend/outputs`, removed when the prediction is deleted). Supported formats: WAV, MP3,
FLAC (M4A/AAC is not decodable by the backend and is not accepted).

---

# Docker

Build:

```bash
docker build -t aerosense-backend .
```

Run:

```bash
docker run -p 8000:8000 aerosense-backend
```

The Docker image includes the trained model checkpoints required for
inference.

The raw training dataset is excluded because it is only required for
retraining. The container runs as an unprivileged user, stores SQLite in the `/data`
volume (`docker run -v aerosense-data:/data ...`), restricts CORS to
`CORS_ORIGINS` and has a `HEALTHCHECK` on `/api/v1/health`.

**Status:** the Dockerfile has **not been built or run** in the environment where this
was last verified (Docker was not installed). Treat it as unverified until you run
`docker build` and `curl localhost:8000/api/v1/health`. No hosted deployment exists.

---

# Model Registry

The deployed model metadata is maintained in:

```text
models/registry.json
```

The registry records information such as:

* Model name
* Version
* Architecture
* Training metadata
* Feature configuration
* Active status
* Anomaly threshold

This prevents the frontend/backend from presenting untrained architectures
as active production models.

---

# Limitations

AeroSense is a working research/engineering prototype. Its limitations are
explicit.

### 1. Dataset-limited fault taxonomy

The real dataset contains:

```text
normal
1_broken
2_broken
3_4_broken
```

These represent propeller-damage severity.

The system is not currently a general-purpose detector for bearing,
motor, or aircraft-engine faults.

### 2. CNN + Autoencoder only

CRNN and Transformer architectures were not trained to convergence in the
available CPU-only environment.

They are therefore **not reported as trained models**.

### 3. Indoor-flight data

The dataset consists of indoor UAV recordings.

Generalization to:

* Outdoor environments
* Different UAV platforms
* Different microphones
* Different acoustic environments

has not yet been established.

### 4. Noise robustness

Training includes noise augmentation, but a dedicated controlled SNR
robustness study has not yet been completed.

### 5. No experiment tracker

MLflow is not currently integrated.

Training metadata is stored using structured JSON logs and checkpoints.

### 6. Upload-based inference

The current application processes uploaded audio recordings.

Continuous microphone streaming is not currently implemented.

---

# Future Work

* Train and benchmark CRNN architecture
* Train and benchmark Transformer architecture
* Controlled SNR/noise robustness experiments
* Expand the dataset with motor and bearing fault recordings
* Outdoor UAV testing
* Multi-UAV domain evaluation
* Microphone/domain adaptation
* Embedding-space unknown-anomaly detection
* Continuous microphone inference
* Background inference workers
* Experiment tracking
* Production deployment
* Edge-device optimization

---

# Technical Stack

### Machine Learning

* Python
* PyTorch
* NumPy
* SciPy
* Librosa
* scikit-learn

### Backend

* FastAPI
* Pydantic
* SQLAlchemy
* SQLite
* Uvicorn

### Frontend

* React
* TypeScript
* Vite
* Recharts
* Web Audio API

### Engineering

* Pytest
* Docker
* Git
* GitHub
* REST APIs
* OpenAPI / Swagger

---

# Project Status

| Component                       | Status        |
| ------------------------------- | ------------- |
| Real UAV acoustic dataset       | ✅             |
| Leakage-safe preprocessing      | ✅             |
| CNN classifier                  | ✅ Trained     |
| CNN evaluation                  | ✅             |
| Autoencoder                     | ✅ Trained     |
| Anomaly threshold fitting       | ✅             |
| Unknown anomaly decision engine | ✅             |
| Grad-CAM explainability         | ✅             |
| FastAPI backend                 | ✅             |
| Database persistence            | ✅             |
| Analytics API                   | ✅             |
| React frontend                  | ✅             |
| Real API integration            | ✅             |
| Automated tests                 | ✅ 38 passing  |
| Docker support                  | ✅             |
| CRNN                            | ⏳ Future work |
| Transformer                     | ⏳ Future work |
| Continuous audio streaming      | ⏳ Future work |

---

# License & Dataset Attribution

AeroSense uses the:

**TII UAV Propeller Anomaly Audio Dataset**

Technology Innovation Institute, ICASSP 2022.

Dataset repository:

[https://github.com/tiiuae/UAV-Propeller-Anomaly-Audio-Dataset](https://github.com/tiiuae/UAV-Propeller-Anomaly-Audio-Dataset)

Refer to the upstream repository for dataset licensing and attribution
requirements.

---

<div align="center">

### AeroSense

**Acoustic intelligence for UAV health monitoring.**

From raw audio → deep learning → anomaly detection → explainable diagnosis.

</div>

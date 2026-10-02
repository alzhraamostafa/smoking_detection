# 🚭 AI Smoking Detection System

An end-to-end AI-powered smoking detection system designed for CCTV-style surveillance environments.

The system combines **cigarette object detection** with **smoking behavior recognition**, exposes the models through a **FastAPI backend**, and visualizes detected alerts and evidence using **Grafana**.

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │     Video Input     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Camera Worker     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      Backend        │
                    └───────┬─────┬───────┘
                            │     │
                 ┌──────────┘     └──────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ YOLO Cigarette  │         │ ResNet18 + LSTM │
        │    Detector     │         │ Behavior Model  │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 └────────────┬──────────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Alert Generation  │
                    │      + Evidence     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Grafana       │
                    │  Video + Evidence   │
                    └─────────────────────┘
```

## 🤖 AI Models

### 1. Cigarette Detection

A YOLO-based object detection model is used to detect cigarettes in surveillance-style images.

The current model is:

```text
models/best_v3.pt
```

The model was trained on surveillance-oriented data with varying image quality.

### 2. Smoking Behavior Recognition

A video-based behavior model was developed using:

```text
ResNet18 → LSTM → Linear
```

The model receives **16 uniformly sampled video frames**, resized to `224 × 224` and ImageNet-normalized.

Input shape:

```text
[Batch, 16, 3, 224, 224]
```

The behavior classes are:

```text
0 → Not Smoking
1 → Smoking
```

The trained checkpoint is kept locally and is intentionally excluded from GitHub because of its size.

## ⚙️ Backend

The backend is implemented using **FastAPI**.

Main components include:

```text
app/
├── main.py
├── video_utils.py
└── behaviour_model.py
```

The main video detection endpoint is:

```text
POST /detect/clip
```

The system accepts a short video clip, samples frames, performs behavior analysis, and uses the cigarette detector to support smoking detection.

Media endpoints include:

```text
GET /alerts/latest/media
GET /alerts/latest/video
GET /alerts/latest/evidence
```

The media API provides URLs for the latest annotated video and evidence image.

## 📊 Monitoring & Visualization

Grafana is integrated with the FastAPI backend through the **Infinity datasource**.

The current setup displays:

* Annotated smoking-detection video
* Evidence image
* Latest alert media

The Grafana → FastAPI communication uses the Docker service name:

```text
http://api:8000
```

while browser-accessible media URLs use:

```text
http://localhost:8000
```

## 🐳 Docker

The project uses Docker Compose with multiple services, including:

```text
api
camera-worker
grafana
```

Typical local endpoints:

```text
FastAPI  → http://localhost:8000
Grafana  → http://localhost:3000
```

Start the project:

```bash
docker compose up -d
```

Stop the containers without deleting them:

```bash
docker compose stop
```

Start them again:

```bash
docker compose start
```

View logs:

```bash
docker compose logs -f
```

Check container status:

```bash
docker compose ps
```

## 📁 Project Structure

```text
smoking-project/
│
├── app/
│   ├── main.py
│   ├── video_utils.py
│   └── behaviour_model.py
│
├── models/
│   ├── best_v3.pt
│   └── best_behavior_model.pth
│
├── data/
│   └── data.yaml
│
├── grafana/
│
├── videos/
│
├── runs/
│
├── compose.yaml
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

> Model weights, videos, databases, generated runs, and local environments are excluded from version control.

## 📈 Model Evaluation

### Cigarette Detection

Current validation results for the YOLO model:

* Precision: **0.791**
* Recall: **0.696**
* mAP@50: **0.760**
* mAP@50–95: **0.387**

### Behavior Recognition

Test results:

* Accuracy: **89.71%**
* Smoking Precision: **75.00%**
* Smoking Recall: **88.24%**
* Smoking F1-score: **81.08%**

## 🛠️ Technologies

* Python
* PyTorch
* TorchVision
* Ultralytics YOLO
* OpenCV
* FastAPI
* Docker / Docker Compose
* Grafana
* Infinity Datasource
* SQLite

## 🎯 Project Goal

The goal is to build a practical computer-vision pipeline for detecting smoking activity in surveillance environments, combining:

1. Object detection
2. Video-based behavior recognition
3. Backend inference APIs
4. Automated alert generation
5. Evidence generation
6. Monitoring and visualization

The system is designed as an end-to-end AI engineering project rather than only a standalone machine-learning model.

# Architecture and Flow Charts

This document provides visual Mermaid charts for the main project flows.
Each chart is followed by an element-by-element explanation.

## 1) System Overview

```mermaid
flowchart LR
    U[User]
    W[Web UI<br/>templates + static/js]
    A[Flask App<br/>app.py]
    API[API Endpoint<br/>POST /api/predict]
    H[MediaPipe Hands]
    M[Model<br/>models/*.pkl]
    R[Prediction Result]

    U --> W
    U --> API
    W --> A
    API --> A
    A --> H
    H --> M
    M --> R
    R --> W
    R --> U
```

### Element Guide

- `User`: Person using browser UI or sending direct API requests.
- `Web UI`: Frontend pages and JavaScript that capture frames and render outputs.
- `Flask App`: Main backend that coordinates frame decoding, feature extraction, and prediction.
- `API Endpoint`: Programmatic interface for single-image inference without webpage usage.
- `MediaPipe Hands`: Landmark detector used to generate structured hand features.
- `Model`: Trained classifier stored as `.pkl` files in `models/`.
- `Prediction Result`: Final class output and optional quality metrics returned to UI/API caller.

## 2) Training Pipeline

```mermaid
flowchart TD
    C[Collect Images<br/>training/collect_imgs.py]
    D[Create Dataset<br/>training/create_dataset.py]
    F[Filter Invalid Frames<br/>no landmarks / blurry]
    T[Train Classifier<br/>training/train_classifier.py]
    O[Outputs<br/>.pkl + .txt + .json]
    CV[Convert Reports<br/>training/convert_model_reports.py]
    DEP[Deployment Ready<br/>app.py + API]

    C --> D
    D --> F
    F --> T
    T --> O
    O --> CV
    CV --> DEP
```

### Element Guide

- `Collect Images`: Captures class-wise sign images into `data/` folders.
- `Create Dataset`: Converts images into normalized landmark features.
- `Filter Invalid Frames`: Drops poor samples to improve downstream model quality.
- `Train Classifier`: Builds the RandomForest model from feature dataset.
- `Outputs`: Produces model artifact and model-quality reports.
- `Convert Reports`: Converts/normalizes report formats for sidebar and analytics consumption.
- `Deployment Ready`: Makes trained model usable in web and API inference flows.

## 3) API Inference Sequence

```mermaid
sequenceDiagram
    participant Client as API Client
    participant Flask as Flask /api/predict
    participant Proc as process_frame_data
    participant MP as MediaPipe
    participant Model as RandomForest Model

    Client->>Flask: POST image (multipart or base64)
    Flask->>Proc: Decode frame + options
    Proc->>MP: Detect hand landmarks
    MP-->>Proc: Landmark coordinates
    Proc->>Model: Predict class from features
    Model-->>Proc: Predicted class index
    Proc-->>Flask: prediction + metrics
    Flask-->>Client: JSON response
```

### Element Guide

- `API Client`: External app/script calling `/api/predict`.
- `Flask /api/predict`: Request validator and response formatter.
- `process_frame_data`: Shared inference function used by both UI and API paths.
- `MediaPipe`: Produces landmark tensors from incoming image frame.
- `RandomForest Model`: Converts landmark feature vector into class prediction.

## 4) Telemetry and Stats Flow

```mermaid
flowchart LR
    P[Landing Page<br/>docs/index.html]
    JS[landing.js]
    CF[CFlair Counter<br/>counter.vkrishna04.me]
    B[Badge Endpoint]
    S[Stats Widgets]

    P --> JS
    JS -->|POST /api/views/:project| CF
    JS -->|GET /api/views/:project| CF
    CF --> S
    CF --> B
    B --> P
```

### Element Guide

- `Landing Page`: Public docs homepage hosted on GitHub Pages.
- `landing.js`: Client script that increments and fetches project views.
- `CFlair Counter`: External telemetry service used for anonymous count metrics.
- `Badge Endpoint`: Embeddable counter badge image endpoint.
- `Stats Widgets`: On-page values showing views and health state.

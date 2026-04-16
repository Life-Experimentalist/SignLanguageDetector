# Sign Language Detector

Real-time sign language detection with a Flask web UI, model management endpoints, and a training pipeline.

![Project Views](https://counter.vkrishna04.me/api/views/sign-language-detector/badge?style=flat-square&color=brightgreen&label=views)

## Why This Exists (STAR)

### Situation
Sign language accessibility tools are often fragmented between research scripts and production apps, making it hard to train, ship, and test models in one place.

### Task
Provide a practical end-to-end system that supports:
- image collection and dataset generation,
- model training and report conversion,
- real-time browser-based inference with model switching.

### Action
This project implements:
- a training framework under `training/` for dataset and model lifecycle,
- Flask applications (`app.py`, `app_multi_client.py`) for single and multi-client usage,
- shared utilities in `utils/` for consistent processing and configuration,
- model report parsing and UI display for model quality visibility.

### Result
You get a workflow that can move from data collection to live inference quickly, with configurable runtime behavior and reusable scripts for local development.

## Project Structure

```
SignLanguageDetector/
├── app.py
├── app_multi_client.py
├── training/
├── utils/
├── models/
├── templates/
├── static/
└── docs/
```

## Quick Start (4 Steps, uv Recommended)

1. Clone repository

```powershell
git clone https://github.com/Life-Experimentalist/SignLanguageDetector.git
Set-Location SignLanguageDetector
```

2. Create a fresh uv environment

```powershell
uv venv --python 3.12 .venv
```

If your machine only has Python 3.13 installed, uv can still provision Python 3.12 for this project automatically.

3. Install dependencies

```powershell
uv sync --python .venv\Scripts\python.exe
```

4. Run app

```powershell
uv run --python .venv\Scripts\python.exe python app.py
```

Open the app in your browser:

`http://localhost:5000`

## Alternative (requirements.txt)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Documentation

- [Docs Overview](docs/README.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Roadmap](docs/ROADMAP.md)
- [API Reference](docs/API.md)
- [Release Notes](docs/RELEASE_NOTES.md)
- [Scripts and Commands](docs/SCRIPTS.md)
- [Project TODO](docs/TODO.md)
- [Integration Guide](docs/INTEGRATION.md)
- [Telemetry Integration](docs/TELEMETRY.md)

## Useful uv Commands

```powershell
uv run --python .venv\Scripts\python.exe python app.py
uv run --python .venv\Scripts\python.exe python app_multi_client.py
uv run --python .venv\Scripts\python.exe python training_pipeline.py
uv run --python .venv\Scripts\python.exe python training/convert_model_reports.py
```

## Inference API (No Webpage Required)

After starting the server, you can call a direct API endpoint:

- `POST /api/predict`

Supported input formats:

- `multipart/form-data` with file field `image`
- `application/json` with `image_base64`

Optional flags:

- `show_landmarks` (`true`/`false`, default `false`)
- `include_visuals` (`true`/`false`, default `false`)

PowerShell example (multipart upload):

```powershell
$form = @{
  image = Get-Item .\data\0\sample.jpg
  show_landmarks = "false"
  include_visuals = "false"
}
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method Post -Form $form
```

PowerShell example (base64 JSON):

```powershell
$bytes = [System.IO.File]::ReadAllBytes(".\data\0\sample.jpg")
$payload = @{
  image_base64 = [System.Convert]::ToBase64String($bytes)
  show_landmarks = $false
  include_visuals = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method Post -ContentType "application/json" -Body $payload
```

## Telemetry and Stats

This project is linked with [CFlair-Counter](https://github.com/Life-Experimentalist/CFlair-Counter?tab=readme-ov-file#cflair-counter) hosted at `https://counter.vkrishna04.me`.

- Increment endpoint used by this app:
  `POST https://counter.vkrishna04.me/api/views/sign-language-detector`
- Badge endpoint:
  `https://counter.vkrishna04.me/api/views/sign-language-detector/badge?style=flat-square&color=brightgreen&label=views`

## License

Licensed under Apache License 2.0. See [LICENSE.md](LICENSE.md).

## Warning

Anonymized telemetry is collected using the CFlair-Counter project and is used only to display project stats.

Training quality matters: better capture quality produces better predictions. The dataset builder automatically skips images without detectable hand landmarks and discards blurry frames before training.

To disable it, create a `.env` file and set:

```env
DISABLE_ANONYMOUS_TELMETRY=true
```
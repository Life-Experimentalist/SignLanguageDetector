# Sign Language Detector

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Apache%202.0-22c55e)](LICENSE.md)
[![MediaPipe](https://img.shields.io/badge/mediapipe-0.10-ff6f00)](https://mediapipe.dev/)
[![Flask](https://img.shields.io/badge/flask-3.1-000000?logo=flask)](https://flask.palletsprojects.com/)
![Views](https://counter.vkrishna04.me/api/views/sign-language-detector/badge?style=flat-square&color=brightgreen&label=views)

End-to-end sign language recognition — collect images, train a classifier, and serve real-time browser inference with model analytics, all from one project.

## Features

- **Real-time browser inference** — webcam → MediaPipe hand landmarks → RandomForest → prediction, no plugins
- **Interactive training CLI** — guided flow for image collection, dataset creation, training, and reporting
- **Model analytics sidebar** — F1 scores, top/weak classes, accuracy rendered from JSON reports
- **Headless REST API** — `POST /api/predict` accepts file upload or base64 JSON, returns prediction
- **Hot model switching** — swap `.pkl` files from the UI dropdown without restarting the server
- **One-command setup** — `uv run app` provisions Python, creates venv, installs deps, and starts the server

## Quick Start

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) — install once with `pip install uv` or `irm https://astral.sh/uv/install.ps1 | iex` on Windows.

```powershell
git clone https://github.com/Life-Experimentalist/SignLanguageDetector.git
cd SignLanguageDetector
uv run app
```

Open **http://localhost:5000**. `uv run app` auto-syncs the environment on first run — no manual `pip install` needed.

## Commands

All commands run inside the managed uv environment — no activation needed.

| Command | What it does |
|---|---|
| `uv run app` | Start the single-client web server |
| `uv run train` | Interactive training CLI (collect → dataset → train → infer) |
| `uv run convert-reports` | Regenerate JSON model reports from trained `.pkl` files |
| `uv run python app_multi_client.py` | Multi-client server with per-session isolation |

## Training a Model

Run `uv run train` and follow the prompts. Each stage is optional — skip any you've already completed.

| Stage | What happens |
|---|---|
| 1 — Collect images | Captures webcam frames per sign class into `data/<class>/`. Blurry and hand-free frames are rejected. |
| 2 — Create dataset | Extracts normalized MediaPipe landmark coordinates and saves them as a pickle dataset. |
| 3 — Train classifier | Trains RandomForest, writes `models/model.pkl`, `model.txt` report, and `model.json` metrics. |
| 4 — Analyze | Converts text reports to structured JSON. Surfaces per-class F1 scores and accuracy. |
| 5 — Infer | Runs inference against new input to verify the model before deploying. |

Or run individual stages directly:

```powershell
uv run python training/collect_imgs.py
uv run python training/create_dataset.py
uv run python training/train_classifier.py
uv run convert-reports
```

## Inference API

Start the server (`uv run app`), then call `POST /api/predict` from any HTTP client.

**Multipart file upload:**

```powershell
$form = @{
  image          = Get-Item .\data\0\sample.jpg
  show_landmarks = "false"
  include_visuals = "false"
}
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method Post -Form $form
```

**Base64 JSON:**

```powershell
$bytes   = [System.IO.File]::ReadAllBytes(".\data\0\sample.jpg")
$payload = @{
  image_base64   = [System.Convert]::ToBase64String($bytes)
  show_landmarks  = $false
  include_visuals = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" `
  -Method Post -ContentType "application/json" -Body $payload
```

**Response:**

```json
{
  "prediction": "A",
  "brightness": 142.5,
  "contrast": 68.3,
  "low_brightness": false,
  "model": "model.pkl"
}
```

Optional flags: `show_landmarks` (bool, default `false`), `include_visuals` (bool, default `false`).

## Configuration

Copy `.env.example` to `.env` and adjust as needed. All variables have sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Server port |
| `DEBUG` | `False` | Flask debug mode |
| `MODELS_DIR` | `./models` | Path to trained model files |
| `DATA_DIR` | `./data` | Path to image dataset |
| `QUIZ_DURATION` | `30` | Seconds per quiz round |
| `QUIZ_NUM_GUESSES` | `5` | Guesses allowed per round |
| `DEMO_QUIZ_LETTERS` | `ABCDE` | Letters shown in the demo quiz |
| `DISABLE_ANONYMOUS_TELEMETRY` | `false` | Set `true` to opt out of view counter ping |
| `TELEMETRY_COUNTER_BASE_URL` | `https://counter.vkrishna04.me` | Counter service base URL |

## Project Structure

```
SignLanguageDetector/
├── app.py                   # Single-client Flask server
├── app_multi_client.py      # Multi-client Flask server (session isolation)
├── interactive_cli.py       # Training pipeline CLI
├── pyproject.toml           # Dependencies + uv run scripts
├── .env.example             # Environment variable reference
├── training/                # collect_imgs, create_dataset, train_classifier, convert_model_reports
├── utils/                   # config, app_utils, utils, scripts
├── models/                  # Trained .pkl files + .txt/.json reports
├── templates/               # Jinja2 HTML templates (layout, index, quiz, quiz_demo, sidebar)
├── static/css/              # model-info.css
└── docs/                    # GitHub Pages landing site
```

## GitHub Pages

The landing page at `https://sign.vkrishna04.me` is served from the `docs/` folder.

- Entry: `docs/index.html`
- Styles: `docs/styles.css`
- Stats: `docs/landing.js`
- SEO: `docs/robots.txt`, `docs/sitemap.xml`
- Custom domain: `docs/CNAME` → `sign.vkrishna04.me`
- Branding assets (optional): `docs/static/branding/` — see `docs/BRANDING_PROMPTS.md`

## Documentation

| Doc | Description |
|---|---|
| [API Reference](docs/API.md) | All endpoints with request/response examples |
| [Architecture](docs/ARCHITECTURE.md) | System diagrams and data-flow charts |
| [Contributing](docs/CONTRIBUTING.md) | How to open issues and submit PRs |
| [Roadmap](docs/ROADMAP.md) | Planned features and improvements |
| [Scripts Reference](docs/SCRIPTS.md) | Every training and utility script |
| [Integration Guide](docs/INTEGRATION.md) | Embedding the API in external services |
| [Telemetry](docs/TELEMETRY.md) | How the anonymous view counter works |
| [Release Notes](docs/RELEASE_NOTES.md) | Changelog |

## License

Apache License 2.0. See [LICENSE.md](LICENSE.md).

Anonymous telemetry (a single view-counter ping per page load) is sent via [CFlair-Counter](https://github.com/Life-Experimentalist/CFlair-Counter). Set `DISABLE_ANONYMOUS_TELEMETRY=true` in `.env` to opt out. Training quality matters — better capture conditions produce better predictions. The dataset builder automatically skips frames without detectable hand landmarks and frames flagged as too blurry.

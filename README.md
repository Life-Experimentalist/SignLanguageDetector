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

## Quick Start (3 Steps)

1. Clone repository

```powershell
git clone https://github.com/Life-Experimentalist/SignLanguageDetector.git
Set-Location SignLanguageDetector
```

2. Install dependencies (uv auto-provisions Python 3.12 if needed)

```powershell
uv sync
```

3. Run the app

```powershell
uv run app
```

Open in your browser: `http://localhost:5000`

## Alternative (pip + requirements.txt)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Documentation

- [Docs Overview](docs/README.md)
- [Landing Page](docs/index.html)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Roadmap](docs/ROADMAP.md)
- [API Reference](docs/API.md)
- [Architecture Charts](docs/ARCHITECTURE.md)
- [Branding Prompts](docs/BRANDING_PROMPTS.md)
- [Release Notes](docs/RELEASE_NOTES.md)
- [Scripts and Commands](docs/SCRIPTS.md)
- [Project TODO](docs/TODO.md)
- [Integration Guide](docs/INTEGRATION.md)
- [Telemetry Integration](docs/TELEMETRY.md)

## GitHub Pages Setup

The landing page is ready in the `docs/` folder for GitHub Pages.

- Entry file: `docs/index.html`
- Styles: `docs/styles.css`
- Stats integration: `docs/landing.js`
- SEO files: `docs/robots.txt`, `docs/sitemap.xml`
- Custom domain file: `docs/CNAME` (set to `sign.vkrishna04.me`)
- Branding asset target: `docs/static/branding/`

## Available Commands

All commands run inside the managed uv environment automatically — no activation needed.

| Command | What it does |
|---|---|
| `uv run app` | Start the web server (single-client) |
| `uv run train` | Interactive training CLI (collect → dataset → train → infer) |
| `uv run convert-reports` | Regenerate model JSON reports from trained `.pkl` files |
| `uv run python app_multi_client.py` | Start the multi-client web server |

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
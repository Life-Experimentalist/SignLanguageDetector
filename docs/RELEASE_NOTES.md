# Release Notes

## Current Handoff

- Switched model artifacts from `.p` to `.pkl`.
- Added a direct API for image-based inference.
- Kept the browser UI and live camera feed intact.
- Added readable class labels in the sidebar highlights.
- Added automatic rejection of blurry frames during dataset creation.
- Updated the README for the uv workflow, telemetry, and API usage.
- Published repository metadata with a clearer description and topics.

## Runtime Status

- Flask app starts successfully.
- Homepage responds with HTTP 200.
- `/api/predict` returns predictions for uploaded images.

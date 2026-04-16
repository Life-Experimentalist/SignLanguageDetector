@echo off
REM Run the training pipeline directly

echo Starting Sign Language Detector Training Pipeline...
python -c "from interactive_cli import training_pipeline; training_pipeline()"

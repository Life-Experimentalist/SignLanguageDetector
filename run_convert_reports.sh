#!/bin/bash
# Quick script to convert model reports with default settings

echo "Converting model reports using direct model testing..."
python -c "from training.convert_model_reports import convert_existing_model_reports; convert_existing_model_reports(lazy=False)"

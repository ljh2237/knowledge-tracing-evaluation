#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m pytest --cov=data_loader --cov=model_adapter --cov=mock_model --cov=pipeline --cov=main --cov=dataset_converters --cov=generate_sample_data --cov=io_utils --cov-report=term-missing --cov-report=json:qa/coverage.json --junitxml=qa/junit.xml
.venv/bin/ruff check .

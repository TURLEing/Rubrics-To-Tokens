#!/bin/bash
set +x

# Get the project root directory (two levels up from the script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_rubrics2token_pipeline.py --config_path "$CONFIG_PATH" --config_name rubric2token_config_qwen3

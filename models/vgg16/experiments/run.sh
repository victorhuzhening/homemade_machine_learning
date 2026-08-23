
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "Project root: $PROJECT_ROOT"

export PYTHON_PATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "Starting training"
echo "$PYTHON_PATH"

# python models/vgg16/train.py \
#     --experiment_name "${1:-vgg16_baseline}" \
#     --dataset_dir "${2:-data/brain-tumor-mri-deduplicated}" \
#     --output_dir "models/vgg16/experiments/runs" \
#     --learning_rate "${3:-1e-4}" \
#     --batch_size "${4:-32}" \
#     --num_epochs "${5:-100}"

#!/bin/bash
# NOTE: avoid `set -u` here; some conda activation/deactivation hooks
# reference optional vars and can fail with "unbound variable".
set -eo pipefail

# Build a clean conda env for Gemma HF judge.
# Usage:
#   bash LLM_Eval/setup_gemma_judge_conda.sh
#   ENV_NAME=my-env bash LLM_Eval/setup_gemma_judge_conda.sh

ENV_NAME="${ENV_NAME:-gemma-judge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

source /projects/b1170/users/kyx8046/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

echo "[1/5] Recreate env: ${ENV_NAME}"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda create -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}" pip

echo "[2/5] Activate env"
conda activate "${ENV_NAME}"

echo "[3/5] Install PyTorch stack (no pinned versions)"
conda install -y -c pytorch -c nvidia \
  pytorch pytorch-cuda=12.1

echo "[4/5] Install HF + runtime libs (no pinned versions)"
python -m pip install --upgrade --no-cache-dir \
  transformers \
  torch \
  accelerate \
  sentencepiece \
  protobuf \
  pandas \
  tqdm

echo "[5/5] Verify key versions"
python - <<'PY'
import torch
import transformers
import sentencepiece
import google.protobuf
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"sentencepiece={sentencepiece.__version__}")
print(f"protobuf={google.protobuf.__version__}")
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print(f"gemma4_in_config_mapping={'gemma4' in CONFIG_MAPPING.keys()}")
PY

echo "Done. Verify with:"
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import torch,transformers,sentencepiece,google.protobuf; print(torch.__version__, transformers.__version__, sentencepiece.__version__, google.protobuf.__version__)\""
echo "Run job with:"
echo "  CONDA_ENV_NAME=${ENV_NAME} sbatch LLM_Eval/run_hf_llm_judge.sh"

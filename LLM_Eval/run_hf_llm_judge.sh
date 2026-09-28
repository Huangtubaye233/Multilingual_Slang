#!/bin/bash
#SBATCH --job-name=hf-llm-judge
#SBATCH --output=LLM_Eval/output/hf_llm_judge_%j.out
#SBATCH --error=LLM_Eval/output/hf_llm_judge_%j.err
#SBATCH --account=p31502
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kefanyu2026@u.northwestern.edu
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# NOTE: avoid `set -u` because conda's activate hooks may reference
# optional vars and fail with "unbound variable" on some clusters.
set -eo pipefail

module purge

export HF_HOME="/projects/b1170/users/kyx8046/hf-cache"
mkdir -p "${HF_HOME}"

export VLLM_CACHE_ROOT="/projects/b1170/users/kyx8046/vllm-cache"
mkdir -p "${VLLM_CACHE_ROOT}"/{xdg,inductor,triton,vllm_worker,usage}
export XDG_CACHE_HOME="${VLLM_CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${VLLM_CACHE_ROOT}/inductor"
export TRITON_CACHE_DIR="${VLLM_CACHE_ROOT}/triton"
export VLLM_WORKER_CACHE_DIR="${VLLM_CACHE_ROOT}/vllm_worker"
export VLLM_USAGE_STATS_LOG_PATH="${VLLM_CACHE_ROOT}/usage/vllm_usage_stats.jsonl"
export TRANSFORMERS_CACHE="${VLLM_CACHE_ROOT}/xdg/transformers"

mkdir -p LLM_Eval/output

source /projects/b1170/users/kyx8046/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gemma-judge}"
conda activate "${CONDA_ENV_NAME}"

# NOTE:
# Do NOT run pip upgrades here. Reinstalling packages at job runtime can
# silently break binary compatibility (e.g., numpy/scipy/torch mismatch).
# Please pre-install dependencies once in the conda environment.

python - <<'PY'
import sys
import torch
import transformers
import tokenizers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

print(f"[env] torch={torch.__version__}")
print(f"[env] transformers={transformers.__version__}")
print(f"[env] tokenizers={tokenizers.__version__}")
print(f"[env] gemma4_supported={'gemma4' in CONFIG_MAPPING.keys()}")
PY

MODEL_ID="${1:-google/gemma-4-E4B-it}"
shift 1 || true

DEFAULT_INPUTS=(
  "LLM_Eval/allenai_Olmo-3-7B-Instruct_hf_interpretation_mono_0shots.csv"
)

INPUT_FILES=()
EXTRA_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == *.csv ]]; then
    INPUT_FILES+=("$arg")
  else
    EXTRA_ARGS+=("$arg")
  fi
done

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  INPUT_FILES=("${DEFAULT_INPUTS[@]}")
fi

NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

python LLM_Eval/grade_interpretation_llm_judge_hf.py \
  --model-id "${MODEL_ID}" \
  --output-prefix "Gemma_graded_" \
  --output-dir "LLM_Eval/Results" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --input-files "${INPUT_FILES[@]}" \
  "${EXTRA_ARGS[@]}"

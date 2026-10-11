#!/bin/bash
#SBATCH --job-name=cossim-sampling
#SBATCH --output=LLM_Eval/output/cossim_sampling_%j.out
#SBATCH --error=LLM_Eval/output/cossim_sampling_%j.err
#SBATCH --account=p31502
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kefanyu2026@u.northwestern.edu
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

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
conda activate slanggen

MODEL_NAME="${1:-sentence-transformers/paraphrase-multilingual-mpnet-base-v2}"
shift 1 || true

THRESHOLD="${THRESHOLD:-0.4}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"

python LLM_Eval/sample_distractors_cossim.py \
  --model-name "${MODEL_NAME}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  "$@"

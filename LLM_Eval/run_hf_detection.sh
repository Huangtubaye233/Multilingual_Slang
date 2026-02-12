#!/bin/bash

#SBATCH --job-name=hf-detect
#SBATCH --output=LLM_Eval/output/hf_detect_%j.out
#SBATCH --error=LLM_Eval/output/hf_detect_%j.err
#SBATCH --account=p31502
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kefanyu2026@u.northwestern.edu
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00

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

MODEL_ID="${1:-allenai/Olmo-3-7B-Think}"
TASK="${2:-both}"
ENGINE="hf"
if [[ $# -ge 3 && "$3" != -* ]]; then
  ENGINE="$3"
  shift 3
else
  shift 2
fi

if [[ $# -gt 0 ]]; then
  python LLM_Eval/run_hf_detection.py \
    --model-id "${MODEL_ID}" \
    --task "${TASK}" \
    --engine "${ENGINE}" \
    "$@"
else
  python LLM_Eval/run_hf_detection.py \
    --model-id "${MODEL_ID}" \
    --task "${TASK}" \
    --engine "${ENGINE}"
fi


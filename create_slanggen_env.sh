#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash recreate_slanggen_env.sh
#   bash recreate_slanggen_env.sh myenv
#   FORCE_RECREATE=1 bash recreate_slanggen_env.sh

ENV_NAME="${1:-slanggen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.slanggen.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

# Prefer user-provided conda root;
CONDA_ROOT=""
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Cannot find conda init script at: ${CONDA_SH}" >&2
  echo "Set CONDA_ROOT and retry, e.g. CONDA_ROOT=\$HOME/miniconda3" >&2
  exit 1
fi

source "${CONDA_SH}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${FORCE_RECREATE:-0}" == "1" ]]; then
    echo "Removing existing env: ${ENV_NAME}"
    conda env remove -y -n "${ENV_NAME}"
  else
    echo "Env ${ENV_NAME} already exists. Set FORCE_RECREATE=1 to replace it." >&2
    exit 1
  fi
fi

echo "Creating env ${ENV_NAME} from ${ENV_FILE}"
conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"

conda activate "${ENV_NAME}"
python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as exc:
    print("Torch check failed:", exc)
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"

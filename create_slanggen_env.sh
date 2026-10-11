#!/usr/bin/env bash
# Conda activation hooks can reference unset variables.
set -eo pipefail

# Usage:
#   bash create_slanggen_env.sh
#   bash create_slanggen_env.sh myenv
#   FORCE_RECREATE=1 bash create_slanggen_env.sh

ENV_NAME="${1:-slanggen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.slanggen.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

# Prefer a user-provided root; otherwise discover the installed Conda.
if [[ -z "${CONDA_ROOT:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Set CONDA_ROOT to your Conda installation directory." >&2
    exit 1
  fi
  CONDA_ROOT="$(conda info --base)"
fi
# Conda on Windows reports backslashes; Bash accepts the forward-slash form.
CONDA_ROOT="${CONDA_ROOT//\\//}"
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
python -m nltk.downloader -q stopwords
python -m pip check
python - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQ_FILE="${1:-${PROJECT_ROOT}/slanggen/requirements.txt}"
ENV_NAME="${ENV_NAME:-slanggen}"
PY_VER="${PY_VER:-3.10}"
INSTALL_PYTORCH="${INSTALL_PYTORCH:-1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
MANUAL_PYTORCH_PACKAGES="${MANUAL_PYTORCH_PACKAGES:-torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

CONDA_REQ="${TMP_DIR}/conda-requirements.txt"
PIP_REQ="${TMP_DIR}/pip-requirements.txt"

python - "${REQ_FILE}" "${CONDA_REQ}" "${PIP_REQ}" <<'PY'
import re
import sys
from collections import OrderedDict
from pathlib import Path

req_path = Path(sys.argv[1])
conda_path = Path(sys.argv[2])
pip_path = Path(sys.argv[3])

conda_lines = []
pip_lines = []
skip_conda = {"cpuonly", "pytorch-mutex", "libtorch"}
manual_pip = {"torch", "torchvision", "torchaudio"}

def normalize_conda_line(raw_line: str) -> str:
    parts = raw_line.split("=")
    if len(parts) <= 2:
        return raw_line
    name, version = parts[0], parts[1]
    if not version:
        return name
    return f"{name}={version}"

def add_conda(spec: str) -> None:
    spec = spec.strip()
    if spec:
        conda_lines.append(spec)

def add_pip(spec: str) -> None:
    spec = spec.strip()
    if spec:
        pip_lines.append(spec)

section = None

for raw in req_path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue

    lowered = line.lower()
    if lowered == "[conda]":
        section = "conda"
        continue
    if lowered == "[pip]":
        section = "pip"
        continue

    name = re.split(r"[=<>!]", line, 1)[0].strip()

    if name in skip_conda or name in manual_pip:
        continue

    if section == "conda":
        add_conda(line)
        continue
    if section == "pip":
        add_pip(line.replace("=pypi_0", ""))
        continue

    if line.endswith("=pypi_0"):
        parts = line.split("=")
        if len(parts) >= 2:
            version = parts[1] if len(parts) > 1 else ""
            add_pip(f"{parts[0]}=={version}" if version else parts[0])
        continue

    add_conda(normalize_conda_line(line))

def dedupe(items):
    return list(OrderedDict.fromkeys(items))

conda_unique = dedupe(conda_lines)
pip_unique = dedupe(pip_lines)

conda_path.write_text(("\n".join(conda_unique) + "\n") if conda_unique else "")
pip_path.write_text(("\n".join(pip_unique) + "\n") if pip_unique else "")
PY

mapfile -t CONDA_PACKAGES < <(sed -e '/^\s*$/d' "${CONDA_REQ}")

echo "Creating conda environment '${ENV_NAME}' (python=${PY_VER})..."
conda create -y -n "${ENV_NAME}" "python=${PY_VER}" "${CONDA_PACKAGES[@]}"

echo "Installing pip-only packages into '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

if [[ "${INSTALL_PYTORCH}" == "1" && -n "${MANUAL_PYTORCH_PACKAGES// }" ]]; then
  echo "Installing PyTorch stack (${MANUAL_PYTORCH_PACKAGES})..."
  if [[ -n "${PYTORCH_INDEX_URL}" ]]; then
    conda run -n "${ENV_NAME}" python -m pip install ${MANUAL_PYTORCH_PACKAGES} --index-url "${PYTORCH_INDEX_URL}"
  else
    conda run -n "${ENV_NAME}" python -m pip install ${MANUAL_PYTORCH_PACKAGES}
  fi
fi

if [[ -s "${PIP_REQ}" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install -r "${PIP_REQ}"
else
  echo "No additional pip packages listed."
fi

echo
echo "Done. Activate via: conda activate ${ENV_NAME}"


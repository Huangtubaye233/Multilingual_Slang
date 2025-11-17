#!/usr/bin/env bash
# setup_slanggen_env_gpu.sh
set -e
ENV_NAME="Slanggen"
PY_VER="3.10"

conda create -y -n "${ENV_NAME}" python=${PY_VER}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -U pip setuptools wheel

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install \
  numpy==1.26.4 scipy==1.10.1 pandas==2.3.0 matplotlib==3.10.5 seaborn==0.13.2 \
  scikit-learn==1.7.0 tqdm nltk==3.9.1 gensim==4.3.2 \
  transformers==4.36.2 tokenizers==0.15.2 sentencepiece==0.2.0 \
  sentence-transformers==2.6.1 \
  huggingface-hub \
  ipywidgets

# Register Jupyter kernel for this conda env
pip install ipykernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import nltk; nltk.download('stopwords'); print("NLTK stopwords downloaded.")
PY

echo "Done. conda activate ${ENV_NAME}"
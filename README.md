# Multilingual Slang Generation

This repository contains code and data utilities for multilingual slang generation,
interpretation, and embedding-based analysis.

## Project Layout

- `slanggen/`: code for representational level training.
- `LLM_Eval/`: behavioral level LLM evaluation scripts with datasets.
- `Data_Collection/`: data collection (of Russian).

## Setup (Conda)

### 1) Create environment (one command)

From the repository root:

```bash
bash create_slanggen_env.sh
```

Optional:

- Recreate existing env:
  ```bash
  FORCE_RECREATE=1 bash create_slanggen_env.sh
  ```

### 2) Activate environment

```bash
conda activate slanggen
```

### 3) Environment spec

The environment definition is tracked in:

- `environment.slanggen.yml`

## Quick Start

### Slang demo / training workflow

```bash
cd slanggen/Demo
jupyter lab
```

### LLM evaluation scripts

```bash
cd LLM_Eval
python run_hf_generation.py --help
python run_hf_interpretation.py --help
python run_gpt_generation.py --help
python run_gpt_interpretation.py --help
```

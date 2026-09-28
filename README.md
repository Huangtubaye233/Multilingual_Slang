# Multilingual Slang Generation

This repository contains code and data utilities for multilingual slang generation,
interpretation, and embedding-based analysis.

## Project Layout

- `slanggen/`: code for representational level training.
- `LLM_Eval/`: behavioral level LLM evaluation scripts with datasets.
- `Data_Collection/`: data collection (of Russian).
- `CatGO/`: vendored categorization library, including local changes. Its files
  are tracked directly in this repository; no submodule checkout is required.
  Source: https://github.com/zhewei-sun/CatGO at
  `47fcf6732d3d92851bb950e287ac1ba85a4425d2`. The original MIT license is retained
  in `CatGO/LICENSE`.

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

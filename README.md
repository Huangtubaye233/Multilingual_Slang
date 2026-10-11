# Multilingual Slang Generation

This repository contains code and data utilities for multilingual slang generation,
interpretation, and embedding-based analysis.

## Project Layout

- `slanggen/`: representational training code, Python runners, and experiment configurations.
- `LLM_Eval/`: behavioral level LLM evaluation scripts with datasets.
- `Data_Collection/`: data collection utilities and reviewer subsets
  (`English_subset.csv`, `Chinese_subset.csv`, and `Russian_subset.csv`, 100 entries each).
- `CatGO/`: vendored categorization library, including local changes. Its files
  are tracked directly in this repository; no submodule checkout is required.
  Source: https://github.com/zhewei-sun/CatGO at
  `47fcf6732d3d92851bb950e287ac1ba85a4425d2`. The original MIT license is retained
  in `CatGO/LICENSE`.

## Setup (Conda)

### 1) Create environment

From the repository root:

```powershell
conda env create -f environment.slanggen.yml
conda run -n slanggen python -m nltk.downloader -q stopwords
conda run -n slanggen python -m pip check
```

Alternatively, in Bash, `bash create_slanggen_env.sh` creates the environment,
downloads NLTK stopwords, and checks dependencies.

To recreate an existing environment with the Bash helper:

```bash
FORCE_RECREATE=1 bash create_slanggen_env.sh
```

### 2) Activate environment

```bash
conda activate slanggen
```

### 3) Environment spec

`environment.slanggen.yml` is the shared environment for representational
experiments, overlap analysis, GPT evaluation, and Hugging Face evaluation with
`--engine hf` (the default). 

## Quick Start

### Representational experiments

Run experiments from the repository root with
[`run_representational.py`](slanggen/Demo/run_representational.py) and
[`representational_config.json`](slanggen/Demo/representational_config.json).
The configuration defines the datasets, training conditions, evaluation targets,
model, contrastive training parameters, and random seeds.

The default configuration uses the size-controlled data in
`slanggen/Demo/size_matched_indices/`. Each condition has 690 entries, with
552/34/104 train/dev/test entries. It includes 13 training conditions covering
single-language and mixed-language data, with English-translated and original
definitions.

#### Prepare inputs

Provide the conventional/slang NPY arrays and their train/dev/test index files
specified by `datasets`, `data_dir`, and `index_dir` in the configuration.
Training also requires the word embedding cache specified by
`word_embedding_cache` (by default, `slanggen/Demo/ft_embed_cache_Urban.pickle`).

To rebuild the size-controlled inputs from the full local `English.csv`,
`Chinese.csv`, and `Russian.csv` in `Data_Collection/`:

```bash
python slanggen/Demo/build_size_control_data.py
```

The builder writes 65 NPY files, selection provenance, a manifest, and
`representational_config_size_control.json` into the output directory.

#### Validate and run

Run all configured experiments:

```bash
python slanggen/Demo/run_representational.py --config slanggen/Demo/representational_config.json
```

For a single condition and one seed, select an experiment ID from the configuration:

```bash
python slanggen/Demo/run_representational.py --config slanggen/Demo/representational_config.json --experiments EN_english --n-runs 1
```

Edit `embed_name`, `training_mode`, `categorizer`, `contrastive_params`, `n_runs`,
and `base_seed` in the JSON to change experiment settings. Additional CLI options
are listed by `python slanggen/Demo/run_representational.py --help`.

#### Results

Results are under `output_dir` (by default,
`slanggen/Demo/representational_results/`). `test_roc_summary.csv` records individual test ROC scores;
`test_roc_aggregate.csv` reports their mean and population standard deviation
across runs. Both summaries are also saved as JSON.

### LLM evaluation scripts

```bash
cd LLM_Eval
python run_hf_generation.py --help
python run_hf_interpretation.py --help
python run_gpt_generation.py --help
python run_gpt_interpretation.py --help
```

For GPT generation/interpretation, set `OPENAI_API_KEY` or use
`--api-key-file` with a plain-text key file. The default key file is
`LLM_Eval/openai_api_key.txt` and is Git-ignored.

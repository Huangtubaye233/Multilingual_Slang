
import argparse
import csv
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np

from representational_data import (
    DEMO_DIR, bootstrap_legacy_imports, dataset_stats, exact_test_overlap,
    load_legacy_dataset,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, config_path):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def check_catgo_dependency():
    """Check the local CatGO used indirectly by SlangGenModel."""
    from inspect import getfile
    from CatGO.categorize import Categorizer

    source = Path(getfile(Categorizer)).resolve()
    expected = DEMO_DIR.parents[1] / "CatGO" / "categorize.py"
    if source != expected.resolve():
        raise ImportError(f"Expected repository CatGO at {expected}, found {source}")
    print(f"CatGO dependency: {source}", flush=True)
    return source.as_posix()


def read_config(path, experiment_ids=None, n_runs_override=None):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = ("data_dir", "index_dir", "output_dir", "datasets", "experiments",
                "embed_name", "training_mode", "categorizer", "n_runs", "base_seed")
    for key in required:
        if key not in config:
            raise ValueError(f"Missing configuration field: {key}")
    if config["training_mode"] not in ("head", "whole"):
        raise ValueError("training_mode must be head or whole")
    normalize_triplets = (config.get("contrastive_params") or {}).get("normalize_triplet_embeddings", False)
    if type(normalize_triplets) is not bool:
        raise ValueError("normalize_triplet_embeddings must be a boolean")
    if normalize_triplets and config["training_mode"] != "head":
        raise ValueError("normalize_triplet_embeddings is currently supported only for training_mode=head")
    params = config.get("contrastive_params") or {}
    resample = params.get("resample_negatives_each_epoch", False)
    weight = params.get("neighbor_loss_weight", 1.0)
    if type(resample) is not bool:
        raise ValueError("resample_negatives_each_epoch must be a boolean")
    if type(weight) not in (int, float) or not np.isfinite(weight) or weight < 0:
        raise ValueError("neighbor_loss_weight must be a finite nonnegative number")
    if config["training_mode"] != "head" and (resample or weight != 1.0):
        raise ValueError("Triplet resampling and weighting are currently supported only for training_mode=head")
    if config["training_mode"] == "whole" and (config.get("contrastive_params") or {}).get("checkpoint_metric") == "dev_roc":
        raise ValueError("dev_roc checkpoint selection is currently supported only for training_mode=head")
    if config["embed_name"] not in (
        "SBERT_mpnet", "SBERT_LaBSE", "SBERT_e5_base", "SBERT_e5_large",
        "SBERT_contrastive", "SBERT_t5", "SBERT-multilingual-MiniLM-L12-v2",
    ):
        raise ValueError("Unsupported embed_name")
    if n_runs_override is not None:
        config["n_runs"] = n_runs_override
    if type(config["n_runs"]) is not int or config["n_runs"] < 1:
        raise ValueError("n_runs must be a positive integer")
    if type(config["base_seed"]) is not int or not 0 <= config["base_seed"] < 2**32:
        raise ValueError("base_seed must be an integer in [0, 2**32)")
    if config["base_seed"] + config["n_runs"] > 2**32:
        raise ValueError("Generated seeds exceed the NumPy seed range")
    config["seeds"] = list(range(config["base_seed"], config["base_seed"] + config["n_runs"]))
    experiments = config["experiments"]
    ids = [exp["id"] for exp in experiments]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate experiment IDs")
    if experiment_ids:
        unknown = set(experiment_ids) - set(ids)
        if unknown:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}")
        experiments = [exp for exp in experiments if exp["id"] in experiment_ids]
    if not experiments:
        raise ValueError("No experiments selected")
    for exp in experiments:
        if not exp["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in exp["id"]):
            raise ValueError(f"Invalid experiment ID: {exp['id']}")
        if not exp["evaluate"] or not set(exp["evaluate"]).issubset({"EN", "ZH", "RU"}):
            raise ValueError(f"{exp['id']}: evaluate must contain a nonempty subset of EN, ZH and RU")
        for dataset_id in [exp["train"], *exp["evaluate"].values()]:
            if dataset_id not in config["datasets"]:
                raise ValueError(f"Unknown dataset: {dataset_id}")
    config["experiments"] = experiments
    return config


def validate_data_configuration(config):
    missing = [key for key in ("data_dir", "index_dir")
               if not isinstance(config[key], str) or not config[key].strip()]
    dataset_ids = dict.fromkeys(dataset_id for exp in config["experiments"]
                               for dataset_id in [exp["train"], *exp["evaluate"].values()])
    for dataset_id in dataset_ids:
        spec = config["datasets"][dataset_id]
        for key in ("conv", "slang", "index_suffix"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                missing.append(f"datasets.{dataset_id}.{key}")
    if missing:
        raise ValueError("Data paths are not configured yet. Fill these fields after preparing "
                         "the clean NPYs and indices: " + ", ".join(missing))


def aggregate_metrics(rows):
    """Summarize scalar test ROC across completed runs of each condition/language."""
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("experiment", "language", "evaluation_dataset",
                                          "train_dataset", "embed_name", "categorizer"))
        groups.setdefault(key, []).append(row)
    aggregates = []
    for group in groups.values():
        values = np.asarray([row["test_roc"] for row in group], dtype=np.float64)
        first = group[0]
        aggregates.append({
            **{name: first[name] for name in ("experiment", "language", "evaluation_dataset",
                                             "train_dataset", "embed_name", "categorizer")},
            "n_runs": len(group), "seeds": ",".join(str(row["seed"]) for row in group),
            "test_roc_mean": float(np.mean(values)),
            "test_roc_std": float(np.std(values, ddof=0)), "std_ddof": 0,
        })
    return aggregates


def save_summaries(output, rows):
    for name, records in (("test_roc_summary", rows),
                          ("test_roc_aggregate", aggregate_metrics(rows))):
        write_json(output / f"{name}.json", records)
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def prepare(config, config_path, data_override=None, index_override=None):
    data_dir = resolve_path(data_override or config["data_dir"], config_path)
    index_dir = resolve_path(index_override or config["index_dir"], config_path)
    loaded, statistics, hashes = {}, {}, {}
    dataset_ids = dict.fromkeys(
        dataset_id for exp in config["experiments"]
        for dataset_id in [exp["train"], *exp["evaluate"].values()])
    for dataset_id in dataset_ids:
        spec = config["datasets"][dataset_id]
        dataset, indices, paths = load_legacy_dataset(spec, data_dir, index_dir)
        loaded[dataset_id] = (dataset, indices)
        statistics[dataset_id] = dataset_stats(dataset, indices)
        if statistics[dataset_id]["unassigned_rows"]:
            print(f"NOTE {dataset_id}: {statistics[dataset_id]['unassigned_rows']} filtered "
                  "rows have no split assignment; verify this is intentional.", flush=True)
        for path in paths.values():
            key = path.as_posix()
            if key not in hashes:
                hashes[key] = file_hash(path)
        print(f"{dataset_id}: {statistics[dataset_id]}", flush=True)
    overlaps = []
    for exp in config["experiments"]:
        training, train_indices = loaded[exp["train"]]
        for language, dataset_id in exp["evaluate"].items():
            target, target_indices = loaded[dataset_id]
            count = exact_test_overlap(training, train_indices, target, target_indices)
            overlaps.append({"experiment": exp["id"], "language": language,
                             "exact_train_test_overlap_rows": count})
            if count:
                print(f"WARNING {exp['id']} -> {language}: {count} test definitions "
                      "also occur in contrastive training. Existing indices are preserved.",
                      flush=True)
    return loaded, {"datasets": statistics, "input_sha256": hashes,
                    "exact_train_test_overlap": overlaps}


def release_gpu(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def link_checkpoint(source, destination):
    # Read-only evaluation checkpoints can share storage within the same run.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def run_one(config, experiment, seed, loaded, directory, trainer_class=None, checkpoint=None,
            run_number=1, on_stage=None):
    import torch
    from slanggen.contrastive import SlangGenTrainer
    from slanggen.encoder import FTCachedEncoder
    from slanggen.model import SlangGenModel
    from slanggen.util import get_rankings, get_roc

    trainer_class = trainer_class or SlangGenTrainer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params = {"embed_name": config["embed_name"], "out_name": "predictions",
              "model": experiment.get("categorizer", config["categorizer"]),
              "prior": None, "prior_name": "uniform",
              "contr_params": config.get("contrastive_params")}
    # Keep the notebook defaults unless an explicit override was supplied.
    if params["contr_params"] is not None:
        defaults = {"train_batch_size": 8, "num_epochs": 4,
                    "triplet_margin": 1, "outpath": config["embed_name"],
                    "head_dim": 768}
        if config["training_mode"] == "whole":
            defaults.update(sbert_lr=5e-4, head_lr=1e-4)
        defaults.update(params["contr_params"])
        defaults["outpath"] = config["embed_name"]
        params["contr_params"] = defaults
    encoder = FTCachedEncoder(config["word_embedding_cache"])
    if on_stage:
        on_stage("contrastive" if checkpoint is None else "checkpoint reuse")
    if checkpoint is None:
        dataset, indices = loaded[experiment["train"]]
        train_root = directory / "contrastive"
        trainer = trainer_class(dataset, word_encoder=encoder,
                                out_dir=train_root.as_posix(), verbose=True)
        model = SlangGenModel(trainer, data_dir=train_root.as_posix(),
                              embed_name=config["embed_name"])
        # Constructing a fixed word encoder can consume RNG state. Reset here
        # so cache hits and the selected condition order do not change training.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.train_contrastive(indices, fold_name="train", params=params,
                                mode=config["training_mode"])
        suffix = "with_head" if config["training_mode"] == "head" else "whole_finetuned"
        checkpoint = train_root / "train" / "SBERT_data" / f"{config['embed_name']}_{suffix}.pt"
        del model, trainer
        release_gpu(torch)
    else:
        print(f"Reusing the identical training dataset/seed checkpoint: {checkpoint}", flush=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    write_json(directory / "training_checkpoint.json", {"path": checkpoint.as_posix(),
               "train_dataset": experiment["train"], "seed": seed})

    results = []
    for language, dataset_id in experiment["evaluate"].items():
        if on_stage:
            on_stage(f"{language} evaluation")
        print(f"\n[{experiment['id']} / seed {seed}] Evaluating {language} ({dataset_id})", flush=True)
        dataset, indices = loaded[dataset_id]
        eval_root = directory / "evaluation" / language
        checkpoint_dir = eval_root / "test" / "SBERT_data"
        checkpoint_dir.mkdir(parents=True)
        link_checkpoint(checkpoint, checkpoint_dir / checkpoint.name)
        trainer = trainer_class(dataset, word_encoder=encoder,
                                out_dir=eval_root.as_posix(), verbose=True)
        model = SlangGenModel(trainer, data_dir=eval_root.as_posix(),
                              embed_name=config["embed_name"])
        trainer.get_trained_embeddings(indices, fold_name="test",
                                       model_path=config["embed_name"])
        model.train_categorization(indices, fold_name="test", params=params)
        model.predict_testset(indices, fold_name="test", params=params)
        scores = model.get_results(fold_name="test", mode="test", params=params)
        expected = (len(indices.test), dataset.V)
        if scores.shape != expected or not np.isfinite(scores).all():
            raise ValueError(f"Invalid test scores for {language}: {scores.shape}, expected {expected}")
        rankings = get_rankings(scores, indices.test, dataset.vocab_ids)
        roc = get_roc(rankings, dataset.V)
        np.save(eval_root / "test_rankings.npy", rankings)
        np.save(eval_root / "test_roc.npy", roc)
        result = {"experiment": experiment["id"], "run": run_number, "seed": seed,
                  "language": language, "evaluation_dataset": dataset_id,
                  "train_dataset": experiment["train"],
                  "embed_name": config["embed_name"], "categorizer": params["model"],
                  "test_n": len(indices.test), "candidate_words": dataset.V,
                  "test_roc": float(np.mean(roc))}
        write_json(eval_root / "metrics.json", result)
        results.append(result)
        write_json(directory / "metrics.json", results)
        print(f"{language} test ROC: {result['test_roc']:.8f}", flush=True)
        del model, trainer, scores
        release_gpu(torch)
    return results, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEMO_DIR / "representational_config.json")
    parser.add_argument("--experiments", nargs="+", help="Run selected IDs from the config")
    parser.add_argument("--n-runs", type=int, help="Override repetitions per condition (distinct seeds)")
    parser.add_argument("--dry-run", action="store_true", help="Load/validate only; no model downloads or training")
    parser.add_argument("--data-dir", help="Override NPY directory (relative to the config)")
    parser.add_argument("--index-dir", help="Override ALL split files; never fall back to old indices")
    parser.add_argument("--output-dir", help="Override output parent directory")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_config(config_path, args.experiments, args.n_runs)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.index_dir is not None:
        config["index_dir"] = args.index_dir
    validate_data_configuration(config)
    config["data_dir"] = resolve_path(config["data_dir"], config_path).as_posix()
    config["index_dir"] = resolve_path(config["index_dir"], config_path).as_posix()
    bootstrap_legacy_imports()
    catgo_source = check_catgo_dependency()
    loaded, audit = prepare(config, config_path)
    if args.dry_run:
        print("Dry run complete: data loaded and splits validated; no training or model downloads.")
        return

    import torch
    cache = resolve_path(config.get("word_embedding_cache", "ft_embed_cache_Urban.pickle"), config_path)
    if not cache.is_file():
        raise FileNotFoundError(cache)
    config["word_embedding_cache"] = cache.as_posix()
    audit["input_sha256"][cache.as_posix()] = file_hash(cache)
    output_parent = resolve_path(args.output_dir or config["output_dir"], config_path)
    config["output_dir"] = output_parent.as_posix()
    output = output_parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "data_audit.json", audit)
    write_json(output / "runtime.json", {"torch": torch.__version__,
               "definition_text": "raw", "head_initialization": "identity",
               "definition_prefix": "query: " if config["embed_name"] == "SBERT_e5_base" and config["training_mode"] == "head" else "",
               "resample_negatives_each_epoch": (config.get("contrastive_params") or {}).get(
                   "resample_negatives_each_epoch", False),
               "neighbor_loss_weight": (config.get("contrastive_params") or {}).get(
                   "neighbor_loss_weight", 1.0),
               "normalize_triplet_embeddings": (config.get("contrastive_params") or {}).get(
                   "normalize_triplet_embeddings", False),
               "checkpoint_metric": (config.get("contrastive_params") or {}).get(
                   "checkpoint_metric", "dev_roc" if config["training_mode"] == "head" else "dev_loss"),
               "catgo_source": catgo_source,
               "cuda_available": torch.cuda.is_available(),
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    print(f"Results directory: {output}", flush=True)
    from slanggen.contrastive import SlangGenTrainer

    # Word distances depend only on the ordered vocabulary and the fixed MPNet
    # word encoder in the existing trainer. Reuse them within this process.
    word_distances = {}

    class CachedWordDistanceTrainer(SlangGenTrainer):
        def preprocess_word_dist(self):
            key = tuple(str(word) for word in self.dataset.vocab)
            if key not in word_distances:
                word_distances[key] = super().preprocess_word_dist()
            return word_distances[key].copy()

    summary = []
    checkpoints = {}
    from tqdm import tqdm
    total_runs = len(config["experiments"]) * config["n_runs"]
    with tqdm(total=total_runs, desc="Representational experiments", unit="run",
              dynamic_ncols=True) as progress:
        for experiment in config["experiments"]:
            for run_number, seed in enumerate(config["seeds"], start=1):
                def on_stage(stage):
                    progress.set_postfix(condition=experiment["id"],
                                         repeat=f"{run_number}/{config['n_runs']}",
                                         seed=seed, stage=stage)

                directory = output / experiment["id"] / f"run_{run_number:02d}_seed_{seed}"
                directory.mkdir(parents=True)
                key = (experiment["train"], seed, experiment.get("categorizer", config["categorizer"]))
                metrics, checkpoint = run_one(config, experiment, seed, loaded, directory,
                             trainer_class=CachedWordDistanceTrainer, checkpoint=checkpoints.get(key),
                             run_number=run_number, on_stage=on_stage)
                checkpoints[key] = checkpoint
                summary.extend(metrics)
                save_summaries(output, summary)
                progress.update(1)
    print(f"Complete. Individual runs: {output / 'test_roc_summary.csv'}")
    print(f"Mean/std across runs: {output / 'test_roc_aggregate.csv'}")


if __name__ == "__main__":
    main()

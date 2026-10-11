"""Data and import helpers for the notebook-compatible experiment runner."""

from collections import namedtuple
from pathlib import Path
import pickle
import sys
import types

import numpy as np


DEMO_DIR = Path(__file__).resolve().parent
REPO_DIR = DEMO_DIR.parents[1]
SPLITS = ("train", "dev", "test")
DataIndex = namedtuple("DataIndex", SPLITS)


def bootstrap_legacy_imports():
    """Expose Code as slanggen without the notebooks' Unix-only symlinks."""
    root = str(REPO_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    package = sys.modules.get("slanggen")
    if package is None:
        package = types.ModuleType("slanggen")
        sys.modules["slanggen"] = package
    package.__path__ = [str(REPO_DIR / "slanggen" / "Code")]


class InspectionWord:
    """Attribute container for inspecting old pickles without importing torch."""


class _InspectionUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Word" and module in ("__main__", "slanggen.util"):
            return InspectionWord
        return super().find_class(module, name)


def load_raw_array(path):
    """Read repository-owned NPYs, including pickled __main__.Word objects.

    This loader is for inspection only. Training uses the original dataloaders.
    """
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, dtype = np.lib.format._read_array_header(stream, version)
        if dtype.hasobject:
            array = _InspectionUnpickler(stream).load()
        else:
            stream.seek(0)
            array = np.load(stream, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.shape != shape:
        raise ValueError(f"Invalid array shape in {path}")
    return array


def data_paths(spec, data_dir, index_dir=None):
    missing = [key for key in ("conv", "slang", "index_suffix")
               if not isinstance(spec.get(key), str) or not spec[key].strip()]
    if missing:
        raise ValueError("Dataset paths are not configured yet: " + ", ".join(missing))
    directory = Path(data_dir)
    indices = Path(index_dir) if index_dir is not None else directory
    return {
        "conv": directory / spec["conv"],
        "slang": directory / spec["slang"],
        **{split: indices / f"{split}_ind_{spec['index_suffix']}.npy"
           for split in SPLITS},
    }


def load_indices(paths, n_total=None):
    arrays = []
    for split in SPLITS:
        array = np.load(paths[split], allow_pickle=False)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{paths[split]} must be a 1D integer array")
        if not len(array):
            raise ValueError(f"{paths[split]} is empty")
        if array.min() < 0 or (n_total is not None and array.max() >= n_total):
            raise ValueError(f"{paths[split]} contains out-of-range indices")
        arrays.append(array)
    joined = np.concatenate(arrays)
    if len(np.unique(joined)) != len(joined):
        raise ValueError("Duplicate indices within/across train/dev/test splits")
    return DataIndex(*arrays)


def load_legacy_dataset(spec, data_dir, index_dir=None):
    bootstrap_legacy_imports()
    from slanggen.dataloader import WN_Dataset, OSD_Dataset, ZH_Dataset
    from slanggen.util import Word

    # Conventional NPYs were serialized by notebook cells as __main__.Word.
    setattr(sys.modules["__main__"], "Word", Word)
    paths = data_paths(spec, data_dir, index_dir)
    loaders = {"OSD_Dataset": OSD_Dataset, "ZH_Dataset": ZH_Dataset}
    if spec["loader"] not in loaders:
        raise ValueError(f"Unsupported slang loader: {spec['loader']}")
    conv = WN_Dataset(str(paths["conv"]))
    loader = loaders[spec["loader"]]
    policy = spec.get("filter_policy", "legacy")
    if policy == "frozen_source_selection":
        # Paired text controls retain eligibility fixed on the original text.
        # Re-filtering paraphrases would change row positions and test membership.
        class PreselectedLoader(loader):
            def process_entries(self, slang_path, conv_dataset):
                return self.load_data(slang_path, conv_dataset), conv_dataset.data
        loader = PreselectedLoader
    elif policy != "legacy":
        raise ValueError(f"Unsupported filter_policy: {policy}")
    dataset = loader(str(paths["slang"]), conv)
    indices = load_indices(paths, dataset.N_total)
    return dataset, indices, paths


def dataset_stats(dataset, indices):
    return {
        "filtered_slang_rows": dataset.N_total,
        "candidate_words": dataset.V,
        "candidate_conv_definitions": sum(
            len(dataset.conv_data[word].definitions) for word in dataset.vocab),
        **{split: len(getattr(indices, split)) for split in SPLITS},
        "unassigned_rows": dataset.N_total - sum(map(len, indices)),
    }


def exact_test_overlap(training, train_indices, target, target_indices):
    """Report definition-level overlap; word-level overlap is expected here."""
    train_pairs = {(str(training.slang_data[i].word), training.slang_data[i].def_sent)
                   for i in train_indices.train}
    return sum((str(target.slang_data[i].word), target.slang_data[i].def_sent)
               in train_pairs for i in target_indices.test)

"""Build the confirmed size-control datasets from the original three CSVs.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from representational_data import DEMO_DIR, REPO_DIR, bootstrap_legacy_imports, load_legacy_dataset


BUDGET = 690
LANGUAGES = ("EN", "ZH", "RU")
SPLITS = ("train", "dev", "test")
SOURCE_FILES = {"EN": "English.csv", "ZH": "Chinese.csv", "RU": "Russian.csv"}


@dataclass(frozen=True)
class SourceRow:
    language: str
    csv_row_index: int
    uid: str
    word: str
    conv_english: str
    slang_english: str
    conv_original: str
    slang_original: str

    def definitions(self, representation):
        if representation == "english":
            return self.conv_english, self.slang_english
        return self.conv_original, self.slang_original


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    languages: tuple
    representation: str
    conv: str
    slang: str
    index_suffix: str
    loader: str

    def config_entry(self):
        return {key: getattr(self, key) for key in ("conv", "slang", "index_suffix", "loader")}


def dataset_specs():
    """Keep active Demo filenames; add ZHRU without reusing old Demo arrays."""
    specs = [
        DatasetSpec("EN", ("EN",), "english", "OD_OSD.npy", "OSD_V2.npy", "OD_OSD", "OSD_Dataset"),
        DatasetSpec("ZH", ("ZH",), "english", "ZH_conv_data_chime.npy", "ZH_slang_data_chime.npy", "ZH_chime", "ZH_Dataset"),
        DatasetSpec("ZH_zh", ("ZH",), "original", "ZH_zh_conv_data_chime.npy", "ZH_zh_slang_data_chime.npy", "ZH_zh_chime", "ZH_Dataset"),
        DatasetSpec("RU", ("RU",), "english", "RU_conv_data.npy", "RU_slang_data.npy", "ru", "ZH_Dataset"),
        DatasetSpec("RU_ru", ("RU",), "original", "RU_ru_conv_data.npy", "RU_ru_slang_data.npy", "ru_ru", "ZH_Dataset"),
    ]
    combinations = (
        ("ENZH", ("EN", "ZH"), "ENZH_OD_chime"),
        ("ENRU", ("EN", "RU"), "ENRU_OD"),
        ("ZHRU", ("ZH", "RU"), "ZHRU_chime"),
        ("all", LANGUAGES, "all_OD_chime"),
    )
    for representation, prefix in (("english", "mix"), ("original", "mix_multilingual")):
        for group, languages, suffix in combinations:
            specs.append(DatasetSpec(f"{prefix}_{group}", languages, representation,
                         f"{prefix}_conv_data_{suffix}.npy", f"{prefix}_slang_data_{suffix}.npy",
                         f"{prefix}_{suffix}", "OSD_Dataset"))
    return specs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_slang_definition(text):
    # Match the literal escape cleanup in OSD_Dataset / ZH_Dataset.
    text = re.sub(r"\\x[a-f0-9][a-f0-9]", "", text)
    return re.sub(r"\\(n|t)", "", text)


def read_sources(directory):
    sources, hashes = {}, {}
    for language, filename in SOURCE_FILES.items():
        path = directory / filename
        hashes[filename] = sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"slang", "conv_def", "slang_definition"}
            if language != "EN":
                required |= {"conv_def_en", "slang_definition_en"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"Missing columns in {filename}: {required - set(reader.fieldnames or [])}")
            rows = []
            for index, record in enumerate(reader):
                if any(not (record.get(key) or "").strip() for key in required):
                    raise ValueError(f"Empty required cell in {filename}, data row {index}")
                # Preserve the CSV strings. This ID is tied to the source-file hash
                # plus a zero-based CSV data-row index (not a physical line number).
                content = json.dumps({key: record[key] for key in sorted(required)},
                                     ensure_ascii=False, sort_keys=True)
                uid = f"{language}:{index:06d}:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]}"
                rows.append(SourceRow(language, index, uid, record["slang"],
                    record["conv_def"] if language == "EN" else record["conv_def_en"],
                    record["slang_definition"] if language == "EN" else record["slang_definition_en"],
                    record["conv_def"], record["slang_definition"]))
        # Current clean CSVs have one row per word and no labels shared between
        # languages. Stop if that changes rather than silently merging meanings.
        if len({row.word for row in rows}) != len(rows):
            raise ValueError(f"Duplicate word labels in {filename}; resolve identity before building")
        sources[language] = rows
    for i, first in enumerate(LANGUAGES):
        for second in LANGUAGES[i + 1:]:
            if {row.word for row in sources[first]} & {row.word for row in sources[second]}:
                raise ValueError(f"Shared word labels across {first}/{second}; review before mixing")
    if len(sources["ZH"]) != min(map(len, sources.values())):
        raise ValueError("Chinese is no longer the smallest source dataset")
    if len(sources["ZH"]) // 6 * 6 != BUDGET:
        raise ValueError("The Chinese source count no longer implies a budget of 690")
    return sources, hashes


def joint_eligible_rows(sources, Word, SlangEntry, acronym_check, has_close_conv_def):
    """Align example identities by requiring both variants to survive.

    Use the existing filtering functions, not a new semantic overlap threshold.
    All CSV rows have conventional definitions; vocabulary membership is assured.
    """
    eligible, audit = {}, {}
    for language, rows in sources.items():
        kept_by_representation, reasons = {}, {}
        for representation in ("english", "original"):
            conv_data = {}
            for row in rows:
                conv, _ = row.definitions(representation)
                word = Word(row.word)
                word.attach_def(conv, None, [])
                conv_data[row.word] = word
            kept, dropped = set(), Counter()
            for row in rows:
                _, slang = row.definitions(representation)
                slang = clean_slang_definition(slang)
                entry = SlangEntry(row.word, slang, {})
                if acronym_check(entry):
                    dropped["acronym"] += 1
                elif has_close_conv_def(row.word, slang, conv_data):
                    dropped["lexical_overlap"] += 1
                else:
                    kept.add(row.uid)
            kept_by_representation[representation] = kept
            reasons[representation] = {"eligible": len(kept), "dropped": dict(dropped)}
        joint = kept_by_representation["english"] & kept_by_representation["original"]
        eligible[language] = [row for row in rows if row.uid in joint]
        if len(joint) < BUDGET:
            raise ValueError(f"{language}: only {len(joint)} jointly eligible rows; need 690")
        audit[language] = {"source_rows": len(rows), "joint_eligible": len(joint), **reasons}
    return eligible, audit


def shared_pools_and_splits(eligible, seed):
    """Sample each language's 690-row master pool, then fix its held-out IDs.

    RandomState retains generate_index.py's seeded permutation convention.
    Every language is sampled reproducibly from its own ordered candidate list.
    English/original variants and all mixed conditions share these identities.
    """
    pools, memberships = {}, {}
    for language in LANGUAGES:
        selection = np.random.RandomState(seed).permutation(len(eligible[language]))[:BUDGET]
        pool = [eligible[language][int(index)] for index in selection]
        permutation = np.random.RandomState(seed).permutation(BUDGET)
        n_train, n_dev = int(.8 * BUDGET), int(.05 * BUDGET)
        positions = {
            "train": permutation[:n_train],
            "dev": permutation[n_train:n_train + n_dev],
            "test": permutation[n_train + n_dev:],
        }
        memberships[language] = {
            pool[int(index)].uid: split for split in SPLITS for index in positions[split]}
        pools[language] = pool
    return pools, memberships


def language_quotas(languages):
    # Each condition has exactly 690 rows and global counts 552/34/104.
    # Stratify by language within those counts. For 3 x 230 rows, dev floors
    # sum to 33; give the remaining dev slot to the first language (EN).
    per_language = BUDGET // len(languages)
    quotas = {language: {"train": int(.8 * per_language), "dev": int(.05 * per_language)}
              for language in languages}
    remaining_dev = int(.05 * BUDGET) - sum(q["dev"] for q in quotas.values())
    for language in languages[:remaining_dev]:
        quotas[language]["dev"] += 1
    for quota in quotas.values():
        quota["test"] = per_language - quota["train"] - quota["dev"]
    return quotas


def condition_rows(spec, pools, memberships, seed):
    quotas = language_quotas(spec.languages)
    rows = []
    for language in spec.languages:
        if len(spec.languages) == 1:
            rows.extend(pools[language])
            continue
        for split in SPLITS:
            candidates = [row for row in pools[language] if memberships[language][row.uid] == split]
            selection = np.random.RandomState(seed).permutation(len(candidates))[:quotas[language][split]]
            rows.extend(candidates[int(index)] for index in selection)
    indices = {}
    for split in SPLITS:
        positions = np.asarray([index for index, row in enumerate(rows)
                                if memberships[row.language][row.uid] == split], dtype=np.int64)
        # Shuffle index order while preserving the shared split membership.
        indices[split] = np.random.RandomState(seed).permutation(positions)
    assert len(rows) == len({row.uid for row in rows}) == BUDGET
    assert [len(indices[split]) for split in SPLITS] == [552, 34, 104]
    return rows, indices, quotas


def check_held_out_boundaries(plan):
    for condition in plan.values():
        training_ids = {condition["rows"][int(index)].uid
                        for split in ("train", "dev") for index in condition["indices"][split]}
        for target_id in ("EN", "ZH", "ZH_zh", "RU", "RU_ru"):
            target = plan[target_id]
            test_ids = {target["rows"][int(index)].uid for index in target["indices"]["test"]}
            if training_ids & test_ids:
                raise ValueError("Condition train/dev contains an evaluation test example")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_DIR / "Data_Collection")
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR / "size_matched_indices")
    parser.add_argument("--template-config", type=Path, default=DEMO_DIR / "representational_config.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")

    # This entry point writes files only when explicitly invoked after review.
    # Keep its imports identical to the original dataloaders' utility functions.
    bootstrap_legacy_imports()
    from slanggen.util import Word, SlangEntry, acronym_check, has_close_conv_def

    sources, source_hashes = read_sources(args.source_dir.resolve())
    eligible, source_audit = joint_eligible_rows(
        sources, Word, SlangEntry, acronym_check, has_close_conv_def)
    pools, memberships = shared_pools_and_splits(eligible, args.seed)
    specs = dataset_specs()
    plan = {}
    for spec in specs:
        rows, indices, quotas = condition_rows(spec, pools, memberships, args.seed)
        plan[spec.dataset_id] = {"spec": spec, "rows": rows, "indices": indices, "quotas": quotas}
    check_held_out_boundaries(plan)

    names = [name for spec in specs for name in
             [spec.conv, spec.slang, *(f"{split}_ind_{spec.index_suffix}.npy" for split in SPLITS)]]
    assert len(names) == len(set(names)) == 65
    metadata_names = ["selected_rows.csv", "representational_config_size_control.json", "manifest.json"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in names + metadata_names):
        raise FileExistsError("Build outputs already exist; choose another output directory")

    template = json.loads(args.template_config.read_text(encoding="utf-8-sig"))
    if set(template["datasets"]) != set(plan):
        raise ValueError("Template dataset IDs do not match the 13 planned datasets")
    template["datasets"] = {spec.dataset_id: spec.config_entry() for spec in specs}

    def template_path(field, default):
        path = Path(template.get(field) or default)
        if not path.is_absolute():
            path = args.template_config.resolve().parent / path
        return path.resolve().as_posix()

    template.update(data_dir=".", index_dir=".",
                    word_embedding_cache=template_path("word_embedding_cache", "ft_embed_cache_Urban.pickle"),
                    output_dir=template_path("output_dir", "representational_results"))

    # Stage and reload every output before publishing it. The temporary directory
    # is created inside the explicit output directory and only its own contents
    # are cleaned up by TemporaryDirectory; existing Demo files are never touched.
    with tempfile.TemporaryDirectory(prefix=".size-control-build-", dir=output) as temp:
        staging = Path(temp).resolve()
        assert staging.is_relative_to(output)
        manifest = {"seed": args.seed, "budget": BUDGET, "split_counts": [552, 34, 104],
                    "eligibility": "joint_english_original", "shared_held_out_boundaries": True,
                    "source_sha256": source_hashes, "source_audit": source_audit,
                    "builder_sha256": sha256(__file__),
                    "filter_code_sha256": sha256(REPO_DIR / "slanggen" / "Code" / "util.py"),
                    "numpy_version": np.__version__, "datasets": {}, "npy_sha256": {}}
        provenance = []
        for spec in specs:
            item = plan[spec.dataset_id]
            rows = item["rows"]
            conv = np.empty(BUDGET, dtype=object)
            slang = np.empty((BUDGET, 2), dtype=object)
            for index, row in enumerate(rows):
                conv_text, slang_text = row.definitions(spec.representation)
                word = Word(row.word)
                # The CSVs have no POS tags or usage examples. Store missing POS
                # and an empty sentence list; do not invent linguistic metadata.
                word.attach_def(conv_text, None, [])
                conv[index] = word
                slang[index] = (row.word, slang_text)
                provenance.append({"dataset": spec.dataset_id, "representation": spec.representation,
                    "array_index": index, "uid": row.uid, "language": row.language,
                    "source_file": SOURCE_FILES[row.language], "csv_row_index": row.csv_row_index,
                    "word": row.word, "split": memberships[row.language][row.uid]})
            np.save(staging / spec.conv, conv)
            np.save(staging / spec.slang, slang)
            for split in SPLITS:
                np.save(staging / f"{split}_ind_{spec.index_suffix}.npy", item["indices"][split])
            dataset, indices, _ = load_legacy_dataset(spec.config_entry(), staging, staging)
            if dataset.N_total != BUDGET or dataset.V != BUDGET:
                raise ValueError(f"{spec.dataset_id}: loader changed the 690-example/word budget")
            expected = [(row.word, clean_slang_definition(row.definitions(spec.representation)[1])) for row in rows]
            if [(str(entry.word), entry.def_sent) for entry in dataset.slang_data] != expected:
                raise ValueError(f"{spec.dataset_id}: raw-to-filtered row order changed")
            if any(not np.array_equal(getattr(indices, split), item["indices"][split]) for split in SPLITS):
                raise ValueError(f"{spec.dataset_id}: split round-trip mismatch")
            manifest["datasets"][spec.dataset_id] = {"representation": spec.representation,
                "language_quotas": item["quotas"], "rows": BUDGET, "candidate_words": dataset.V,
                "conv_definitions": BUDGET, **spec.config_entry()}
        manifest["npy_sha256"] = {name: sha256(staging / name) for name in names}
        with (staging / "selected_rows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(provenance[0]))
            writer.writeheader()
            writer.writerows(provenance)
        for name, payload in (("representational_config_size_control.json", template), ("manifest.json", manifest)):
            (staging / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Publish manifest last, after all 65 validated NPYs and the provenance.
        for name in names + metadata_names:
            if (output / name).exists():
                raise FileExistsError(output / name)
            (staging / name).rename(output / name)
    print(f"Built 65 validated NPYs and provenance/configuration files in {output}")
    print("No representational experiment was trained.")


if __name__ == "__main__":
    main()

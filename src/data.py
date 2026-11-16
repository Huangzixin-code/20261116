"""Hugging Face dataset loaders and corpus schema checks."""
from __future__ import annotations

import os

from .config import settings

# Keep Hub-managed files on the project drive too. This must be set before
# importing datasets/huggingface_hub, which read these variables at import time.
os.environ.setdefault("HF_HOME", str(settings.data_cache_dir))
os.environ.setdefault("HF_HUB_CACHE", str(settings.data_cache_dir / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(settings.data_cache_dir / "datasets"))

from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset


def _download_config(local_files_only: bool) -> DownloadConfig | None:
    return DownloadConfig(local_files_only=True) if local_files_only else None


def load_corpus(local_files_only: bool = False) -> Dataset:
    kwargs = {}
    if local_files_only:
        kwargs["download_config"] = _download_config(True)
    dataset = load_dataset(
        settings.corpus_dataset,
        split="train",
        cache_dir=str(settings.data_cache_dir),
        **kwargs,
    )
    required = {"docid", "text", "url"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Corpus missing columns {sorted(missing)}; found {dataset.column_names}")
    return dataset


def load_validation(split: str = "validation", local_files_only: bool = False) -> Dataset:
    kwargs = {}
    if local_files_only:
        kwargs["download_config"] = _download_config(True)
    dataset = load_dataset(
        settings.validation_dataset,
        split=split,
        cache_dir=str(settings.data_cache_dir),
        **kwargs,
    )
    required = {"id", "question", "answer"} if split == "validation" else {"id", "question"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Split {split!r} missing columns {sorted(missing)}; found {dataset.column_names}")
    return dataset


def load_test_dataset(local_files_only: bool = False) -> Dataset:
    """Load the course's blind test questions without requiring gold answers."""
    kwargs = {}
    if local_files_only:
        kwargs["download_config"] = _download_config(True)
    loaded = load_dataset(
        settings.test_dataset,
        cache_dir=str(settings.data_cache_dir),
        **kwargs,
    )
    if isinstance(loaded, DatasetDict):
        if "test" in loaded:
            dataset = loaded["test"]
        elif len(loaded) == 1:
            # Some single-split Hub repositories expose their only split as
            # "train". Use it without assuming the split label.
            dataset = next(iter(loaded.values()))
        else:
            raise ValueError(
                f"Test dataset {settings.test_dataset!r} has splits {list(loaded.keys())}; "
                "expected a 'test' split or exactly one split."
            )
    else:
        dataset = loaded
    required = {"id", "question"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Test dataset {settings.test_dataset!r} is missing {sorted(missing)}; "
            f"found columns {dataset.column_names}."
        )
    # Keep the final benchmark blind even if a future Hub revision publishes
    # an answer-like column alongside the questions.
    answer_columns = [name for name in ("answer", "gold_answer", "gold")
                      if name in dataset.column_names]
    if answer_columns:
        dataset = dataset.remove_columns(answer_columns)
    return dataset


def load_evaluation_dataset(split: str, local_files_only: bool = False) -> Dataset:
    """Select the validation labels or blind course test questions by split."""
    if split == "test":
        return load_test_dataset(local_files_only=local_files_only)
    return load_validation(split, local_files_only=local_files_only)

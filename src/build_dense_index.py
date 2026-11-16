"""Build an optional normalized dense-vector index for the fixed corpus."""
from __future__ import annotations

import argparse

from .config import settings
from .data import load_corpus
from .dense import DenseIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    settings.dense_index_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    print(f"Loaded {len(corpus):,} corpus documents; embedding with {settings.dense_model}.", flush=True)
    DenseIndex.build(corpus, settings.dense_index_dir, settings.dense_model,
                     dataset_name=settings.corpus_dataset,
                     batch_size=args.batch_size)
    print(f"Saved dense index to {settings.dense_index_dir}", flush=True)


if __name__ == "__main__":
    main()

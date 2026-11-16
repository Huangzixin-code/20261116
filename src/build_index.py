"""Build and persist BM25S once; the UI loads the saved index rather than rebuilding it."""
from __future__ import annotations

import json
from pathlib import Path

import bm25s
from dotenv import load_dotenv

from .config import PROJECT_ROOT, settings
from .data import load_corpus


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    settings.data_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_corpus()
    print(f"Loaded {len(dataset):,} corpus rows; fields={dataset.column_names}")
    print("Tokenizing corpus for BM25. This step can use substantial RAM.")
    tokens = bm25s.tokenize(dataset["text"], stopwords="en", show_progress=True)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(tokens, show_progress=True)
    retriever.save(str(settings.index_dir), corpus=None)
    metadata = {"dataset": settings.corpus_dataset, "split": "train",
                "documents": len(dataset), "columns": dataset.column_names,
                "index_method": "BM25S lucene", "stopwords": "en"}
    (settings.index_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved BM25 index and metadata to {settings.index_dir}")


if __name__ == "__main__":
    main()

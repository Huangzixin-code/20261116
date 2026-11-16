"""Compare candidate coverage for sparse, dense, instruction-aware, and hybrid retrieval."""
from __future__ import annotations

import argparse
import json
import os
import re
import string
from pathlib import Path

import bm25s

from .config import PROJECT_ROOT, settings

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from .data import load_corpus, load_validation
from .dense import DenseIndex
from .retrieval import SearchTools


MODES = ("bm25", "dense", "dense_instruct", "hybrid")


def normalize(text: str) -> str:
    text = str(text).casefold().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--mode", choices=MODES, action="append",
                        help="May be repeated; defaults to comparing all four modes.")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.offset < 0 or args.top_k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("offset must be non-negative; top-k and limit must be positive")

    print(f"Loading cached {args.split} labels and corpus...", flush=True)
    dataset = load_validation(args.split, local_files_only=True)
    split_size = len(dataset)
    start = min(args.offset, split_size)
    end = split_size if args.limit is None else min(start + args.limit, split_size)
    questions = dataset.select(range(start, end))
    corpus = load_corpus(local_files_only=True)
    ids = [str(value) for value in corpus["docid"]]
    row_by_id = {doc_id: row for row, doc_id in enumerate(ids)}
    bm25_index = bm25s.BM25.load(str(settings.index_dir), mmap=True)
    modes = args.mode or list(MODES)
    dense_index = None
    if any(mode != "bm25" for mode in modes):
        dense_index = DenseIndex(settings.dense_index_dir, settings.dense_model, ids)

    results: dict[str, dict] = {}
    for mode in modes:
        tools = SearchTools(corpus, bm25_index, args.top_k,
                            retrieval_mode=mode, dense_index=dense_index)
        hits = 0
        reciprocal_ranks = []
        per_question = []
        for item in questions:
            answer = normalize(item["answer"])
            candidates = tools.search(item["question"], top_k=args.top_k)
            found_rank = None
            if answer:
                for rank, candidate in enumerate(candidates, start=1):
                    row = corpus[row_by_id[candidate["id"]]]
                    if answer in normalize(row["text"]):
                        found_rank = rank
                        break
            if found_rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)
            per_question.append({"id": str(item["id"]), "literal_answer_rank": found_rank,
                                 "candidate_ids": [candidate["id"] for candidate in candidates]})
        n = len(questions)
        results[mode] = {
            "count": n,
            "literal_answer_coverage_at_k": hits / n if n else 0.0,
            "literal_answer_coverage_curve": {
                str(k): sum(rank is not None and rank <= k for rank in
                            (item["literal_answer_rank"] for item in per_question)) / n if n else 0.0
                for k in sorted({value for value in (1, 3, 5, 8, 10, 20, 50, 100, args.top_k)
                                 if value <= args.top_k})
            },
            "literal_answer_mrr_at_k": sum(reciprocal_ranks) / n if n else 0.0,
            "top_k": args.top_k,
            "per_question": per_question,
        }
        print(f"{mode}: literal answer coverage@{args.top_k}={hits}/{n}", flush=True)

    summary = {"split": args.split, "evaluation_start": start,
               "evaluation_count": len(questions), "split_size": split_size,
               "note": "Post-hoc literal-answer retrieval diagnostic; not end-to-end answer accuracy.",
               "modes": results}
    output = Path(args.output) if args.output else PROJECT_ROOT / "results" / "retrieval_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote retrieval comparison to {output}", flush=True)


if __name__ == "__main__":
    main()

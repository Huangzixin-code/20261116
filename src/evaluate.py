"""Generate one trajectory-bearing JSONL prediction per dataset question."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import bm25s
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError

from .config import PROJECT_ROOT, settings

# Evaluation uses the already downloaded dataset snapshots. Set offline mode
# before importing datasets/huggingface_hub so legacy dataset scripts do not
# issue Hub HEAD requests on every run.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from .data import load_corpus, load_evaluation_dataset
from .retrieval import create_search_tools
from .agent import SearchAgent


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0,
                        help="Start at this zero-based row in the selected split.")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    # Evaluation reuses the downloaded validation split and corpus. Local-only
    # loading avoids unnecessary Hub HEAD requests and accidental C: cache use.
    dataset_name = settings.test_dataset if args.split == "test" else settings.validation_dataset
    print(f"Loading cached {args.split} dataset ({dataset_name}) from {settings.data_cache_dir}...", flush=True)
    dataset = load_evaluation_dataset(args.split, local_files_only=True)
    split_size = len(dataset)
    evaluation_start = min(args.offset, split_size)
    evaluation_end = split_size if args.limit is None else min(evaluation_start + args.limit, split_size)
    dataset = dataset.select(range(evaluation_start, evaluation_end))
    evaluation_size = len(dataset)
    print(f"Loaded {len(dataset)} {args.split} questions.", flush=True)
    print("Loading cached corpus...", flush=True)
    corpus = load_corpus(local_files_only=True)
    print(f"Loaded {len(corpus)} corpus documents. Loading BM25 index...", flush=True)
    index = bm25s.BM25.load(str(settings.index_dir), mmap=True)
    index_metadata_path = settings.index_dir / "metadata.json"
    index_metadata = (json.loads(index_metadata_path.read_text(encoding="utf-8"))
                      if index_metadata_path.exists() else {})
    print("BM25 index ready.", flush=True)
    search_tools = create_search_tools(corpus, index, top_k=settings.top_k)
    print(f"Retrieval mode: {settings.retrieval_mode}", flush=True)
    print(f"Query fusion mode: {settings.query_fusion_mode}", flush=True)
    agent = SearchAgent(search_tools, progress=True)
    output = Path(args.output or PROJECT_ROOT / "results" / f"{args.split}_predictions.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps({"split": args.split,
                                         "evaluation_start": evaluation_start,
                                         "evaluation_count": evaluation_size,
                                         "split_size": split_size,
                                         "configuration": {
                                             "index_dir": str(settings.index_dir),
                                             "index_method": index_metadata.get("index_method", "unknown"),
                                             "index_parameters": {
                                                 key: index_metadata[key]
                                                 for key in ("k1", "b", "delta")
                                                 if key in index_metadata
                                             },
                                             "retrieval_mode": settings.retrieval_mode,
                                             "query_fusion_mode": settings.query_fusion_mode,
                                             "query_coverage_weight": settings.query_coverage_weight,
                                             "dense_model": settings.dense_model if settings.retrieval_mode != "bm25" else None,
                                             "hybrid_candidate_pool": settings.hybrid_candidate_pool,
                                             "sparse_rrf_weight": settings.sparse_rrf_weight,
                                             "dense_rrf_weight": settings.dense_rrf_weight,
                                             "search_snippet_chars": settings.search_snippet_chars,
                                             "top_k": settings.top_k,
                                             "rrf_k": settings.rrf_k,
                                             "max_search_calls": settings.max_search_calls,
                                             "max_additional_queries": settings.max_additional_queries,
                                             "max_agent_steps": settings.max_agent_steps,
                                             "auto_open_first_candidate": settings.auto_open_first_candidate,
                                             "auto_open_candidate_count": settings.auto_open_candidate_count,
                                             "explicit_candidate_state": settings.explicit_candidate_state,
                                             "structured_evidence_ledger": settings.structured_evidence_ledger,
                                             "min_identity_sources": settings.min_identity_sources,
                                             "strict_final_answer_format": os.getenv(
                                                 "STRICT_FINAL_ANSWER_FORMAT", "false"
                                             ).strip().lower() in {"1", "true", "yes", "on"},
                                         }}, indent=2), encoding="utf-8")
    with output.open("w", encoding="utf-8") as stream:
        for number, item in enumerate(dataset, start=1):
            print(f"Question {number}/{len(dataset)} (id={item['id']}): starting agent...", flush=True)
            try:
                result = agent.run(item["question"])
            except APIStatusError as exc:
                print(f"DeepSeek API returned HTTP {exc.status_code}: {exc.message}", file=sys.stderr)
                if exc.status_code == 401:
                    print("Check DEEPSEEK_API_KEY in D:\\COMP5423\\.env.", file=sys.stderr)
                elif exc.status_code == 402:
                    print("The DeepSeek account may have insufficient balance; check its API console.", file=sys.stderr)
                elif exc.status_code == 404:
                    print("Check DEEPSEEK_MODEL and DEEPSEEK_BASE_URL in D:\\COMP5423\\.env.", file=sys.stderr)
                elif exc.status_code == 429:
                    print("The API rate limit was reached; wait and retry with a smaller --limit.", file=sys.stderr)
                raise SystemExit(1) from None
            except APIConnectionError as exc:
                print(f"Could not connect to DeepSeek API: {exc}", file=sys.stderr)
                print("Check internet access and DEEPSEEK_BASE_URL in D:\\COMP5423\\.env.", file=sys.stderr)
                raise SystemExit(1) from None
            record = {"id": str(item["id"]), "question": item["question"], "answer": result["answer"],
                      "evaluation_index": evaluation_start + number - 1,
                      "evaluation_start": evaluation_start,
                      "evaluation_count": evaluation_size,
                      "split_size": split_size,
                      "trajectory": result["trajectory"], "tool_calls": result["tool_calls"],
                      "total_tokens": result["total_tokens"],
                      "llm_calls": result["llm_calls"],
                      "invalid_action_retries": result["invalid_action_retries"],
                      "stop_reason": result["stop_reason"],
                      "selected_skills": result["selected_skills"]}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{number}/{len(dataset)} id={record['id']} calls={record['tool_calls']} tokens={record['total_tokens']}")
    print(f"Wrote {len(dataset)} predictions to {output}")


if __name__ == "__main__":
    main()

"""Compute normalized exact match and efficiency metrics for validation predictions."""
from __future__ import annotations

import argparse
import json
import os
import re
import string
from pathlib import Path

# Scoring compares against the already cached validation labels and should not
# need network access. Set this before importing Hugging Face datasets.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from .data import load_validation


def normalize(text: str) -> str:
    text = text.casefold().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


ABSTENTION_MARKERS = (
    "insufficient evidence", "cannot verify", "could not verify", "unable to identify",
    "evidence is insufficient", "not enough information", "cannot determine",
)


def record_diagnostics(row: dict) -> dict:
    """Summarize answer support and retrieval actions from one saved trajectory."""
    trajectory = row.get("trajectory", [])
    opened: set[str] = set()
    search_actions = executed_searches = search_fallback_opens = open_fallback_opens = 0
    answer_fallback_opens = 0
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        observation = step.get("observation")
        if isinstance(action, dict):
            if action.get("action") == "search":
                search_actions += 1
                if isinstance(observation, list):
                    executed_searches += 1
            elif action.get("action") == "open" and isinstance(observation, dict):
                if "error" not in observation:
                    doc_id = observation.get("id", action.get("doc_id"))
                    if doc_id is not None:
                        opened.add(str(doc_id))
        if step.get("event") == "invalid_action_fallback_open":
            if step.get("origin_action") == "search":
                search_fallback_opens += 1
            elif step.get("origin_action") == "open":
                open_fallback_opens += 1
            elif step.get("origin_action") == "answer":
                answer_fallback_opens += 1
            if step.get("doc_id") is not None:
                opened.add(str(step["doc_id"]))
        elif step.get("event") in {"search_action_fallback_open", "search_budget_fallback_open"}:
            search_fallback_opens += 1
            if step.get("doc_id") is not None:
                opened.add(str(step["doc_id"]))
        elif step.get("event") in {"auto_open_first_candidate", "auto_open_additional_candidate"}:
            if step.get("doc_id") is not None:
                opened.add(str(step["doc_id"]))

    # Rejected answer attempts have an error observation; only accepted actions
    # represent the final answer used by the evaluator.
    final_action = next((step.get("action") for step in reversed(trajectory)
                         if isinstance(step, dict) and isinstance(step.get("action"), dict)
                         and step["action"].get("action") == "answer"
                         and not (isinstance(step.get("observation"), dict)
                                  and "error" in step["observation"])), {})
    answer = str(final_action.get("answer", row.get("answer", "")))
    citations = [str(value) for value in final_action.get("supporting_doc_ids", [])]
    abstained = any(marker in answer.casefold() for marker in ABSTENTION_MARKERS)
    return {
        "answer": answer,
        "abstained": abstained,
        "citations": citations,
        "opened_doc_count": len(opened),
        "invalid_citation_count": sum(doc_id not in opened for doc_id in citations),
        "search_actions_requested": search_actions,
        "search_calls_executed": executed_searches,
        "search_action_fallback_opens": search_fallback_opens,
        "open_action_fallback_opens": open_fallback_opens,
        "answer_action_fallback_opens": answer_fallback_opens,
        "rejected_search_actions": max(0, search_actions - executed_searches - search_fallback_opens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--details", action="store_true",
                        help="Include per-question correctness and trajectory diagnostics.")
    args = parser.parse_args()
    if args.split == "test":
        parser.error("The course test split is blind and has no local gold labels; do not score it with analyze_results.")
    validation = list(load_validation(args.split, local_files_only=True))
    gold = {str(row["id"]): normalize(row["answer"]) for row in validation}
    records = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata_path = Path(args.predictions).with_suffix(Path(args.predictions).suffix + ".meta.json")
    metadata = (json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists()
                else records[0] if records and "evaluation_start" in records[0] else None)
    if metadata:
        start = int(metadata["evaluation_start"])
        expected_count = int(metadata["evaluation_count"])
        expected_ids = [str(row["id"]) for row in validation[start:start + expected_count]]
    else:
        # Legacy predictions have no declared range; score the unique known IDs present.
        expected_ids = list(dict.fromkeys(str(row["id"]) for row in records if str(row["id"]) in gold))

    by_id = {}
    for row in records:
        if str(row["id"]) in gold:
            by_id[str(row["id"])] = row
    scored = [(by_id[row_id], gold[row_id]) for row_id in expected_ids if row_id in by_id]
    expected_n = len(expected_ids)
    received_n = len(scored)
    correct_n = sum(normalize(row["answer"]) == answer for row, answer in scored)
    diagnostics = [(row, answer, record_diagnostics(row)) for row, answer in scored]
    non_abstaining = [(row, answer, diag) for row, answer, diag in diagnostics if not diag["abstained"]]
    total_citations = sum(len(diag["citations"]) for _, _, diag in diagnostics)
    invalid_citations = sum(diag["invalid_citation_count"] for _, _, diag in diagnostics)
    unknown_ids = {str(row["id"]) for row in records} - set(gold)
    duplicate_ids = len(records) - len({str(row["id"]) for row in records})
    summary = {"count": expected_n,
               "predictions_received": received_n,
               "prediction_coverage": received_n / expected_n if expected_n else 0,
               "normalized_em": correct_n / expected_n if expected_n else 0,
               "normalized_em_on_received": correct_n / received_n if received_n else 0,
               "missing_prediction_ids": [row_id for row_id in expected_ids if row_id not in by_id],
               "duplicate_prediction_rows": duplicate_ids,
               "prediction_ids_missing_gold": len(unknown_ids),
               "evaluation_start": metadata.get("evaluation_start") if metadata else None,
               "split_size": metadata.get("split_size") if metadata else len(validation),
               "mean_tool_calls": sum(row.get("tool_calls", 0) for row, _ in scored) / received_n if received_n else 0,
               "mean_total_tokens": sum(row.get("total_tokens", 0) for row, _ in scored) / received_n if received_n else 0,
               "mean_llm_calls": sum(row.get("llm_calls", 0) for row, _ in scored) / received_n if received_n else 0,
               "invalid_action_retries": sum(row.get("invalid_action_retries", 0) for row, _ in scored),
               "answer_diagnostics": {
                   "abstentions": sum(diag["abstained"] for _, _, diag in diagnostics),
                   "non_abstaining_answers": len(non_abstaining),
                   "non_abstaining_answers_with_citations": sum(bool(diag["citations"])
                                                                  for _, _, diag in non_abstaining),
                   "non_abstaining_answers_without_citations": sum(not diag["citations"]
                                                                     for _, _, diag in non_abstaining),
                   "citation_count": total_citations,
                   "invalid_citation_count": invalid_citations,
                   "citation_validity_rate": ((total_citations - invalid_citations) / total_citations
                                               if total_citations else None),
               },
               "search_diagnostics": {
                   "search_actions_requested": sum(diag["search_actions_requested"]
                                                     for _, _, diag in diagnostics),
                   "search_calls_executed": sum(diag["search_calls_executed"]
                                                 for _, _, diag in diagnostics),
                   "search_action_fallback_opens": sum(diag["search_action_fallback_opens"]
                                                        for _, _, diag in diagnostics),
                   "open_action_fallback_opens": sum(diag["open_action_fallback_opens"]
                                                      for _, _, diag in diagnostics),
                   "answer_action_fallback_opens": sum(diag["answer_action_fallback_opens"]
                                                        for _, _, diag in diagnostics),
                   "rejected_search_actions": sum(diag["rejected_search_actions"]
                                                  for _, _, diag in diagnostics),
               },
               "stop_reasons": {reason: sum(row.get("stop_reason") == reason for row, _ in scored)
                                for reason in sorted({row.get("stop_reason", "unknown") for row, _ in scored})},
               }
    if args.details:
        summary["per_question"] = [
            {"id": str(row["id"]), "correct": normalize(row["answer"]) == answer,
             "gold": next((str(item["answer"]) for item in validation
                           if str(item["id"]) == str(row["id"])), ""),
             **diag, "stop_reason": row.get("stop_reason", "unknown")}
            for row, answer, diag in diagnostics
        ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

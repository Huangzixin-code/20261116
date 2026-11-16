"""Configurable retrieval plus callable document-support tools."""
from __future__ import annotations

from typing import Any

import bm25s
from datasets import Dataset

from .dense import DenseIndex
from .config import settings


class SearchTools:
    def __init__(self, corpus: Dataset, index: bm25s.BM25, top_k: int = 8,
                 retrieval_mode: str = "bm25", dense_index: DenseIndex | None = None):
        self.corpus = corpus
        self.index = index
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode.strip().lower()
        self.dense_index = dense_index
        supported_modes = {"bm25", "dense", "dense_instruct", "hybrid"}
        if self.retrieval_mode not in supported_modes:
            raise ValueError(f"Unsupported RETRIEVAL_MODE {retrieval_mode!r}; choose from {sorted(supported_modes)}.")
        if self.retrieval_mode != "bm25" and self.dense_index is None:
            raise ValueError(f"RETRIEVAL_MODE={self.retrieval_mode} requires a built dense index.")
        self._id_to_row: dict[str, int] | None = None

    def _lookup(self) -> dict[str, int]:
        if self._id_to_row is None:
            self._id_to_row = {str(doc_id): i for i, doc_id in enumerate(self.corpus["docid"])}
        return self._id_to_row

    def _search_bm25(self, query: str, k: int,
                     excluded: set[str]) -> list[dict[str, Any]]:
        query_tokens = bm25s.tokenize([query], stopwords="en")
        candidate_k = min(k + len(excluded) + 20, len(self.corpus))
        doc_indexes, scores = self.index.retrieve(query_tokens, k=candidate_k)
        results = []
        for rank, (row_idx, score) in enumerate(zip(doc_indexes[0], scores[0]), start=1):
            row = self.corpus[int(row_idx)]
            doc_id = str(row["docid"])
            if doc_id in excluded:
                continue
            results.append({"id": doc_id, "score": float(score),
                            "snippet": str(row["text"])[:settings.search_snippet_chars], "url": str(row["url"]),
                            "matched_queries": [query],
                            "query_ranks": [{"query": query, "rank": rank}]})
            if len(results) >= k:
                break
        return results

    def _search_dense(self, query: str, k: int, excluded: set[str],
                      instruction_aware: bool) -> list[dict[str, Any]]:
        assert self.dense_index is not None
        candidates = self.dense_index.search(query, k, instruction_aware, excluded)
        results = []
        for item in candidates:
            row = self.corpus[self._lookup()[item["id"]]]
            results.append({"id": item["id"], "score": item["score"],
                            "snippet": str(row["text"])[:settings.search_snippet_chars], "url": str(row["url"]),
                            "matched_queries": [query],
                            "query_ranks": [{"query": query, "rank": item["rank"]}]})
        return results

    def _rrf_fuse(self, rankings: list[list[dict[str, Any]]], k: int,
                  weights: list[float] | None = None) -> list[dict[str, Any]]:
        weights = weights or [1.0] * len(rankings)
        if len(weights) != len(rankings) or not any(weight > 0 for weight in weights):
            raise ValueError("RRF fusion requires one weight per ranking and at least one positive weight.")
        fused: dict[str, float] = {}
        documents: dict[str, dict[str, Any]] = {}
        provenance: dict[str, list[dict[str, Any]]] = {}
        for source_weight, ranking in zip(weights, rankings):
            for rank, item in enumerate(ranking, start=1):
                doc_id = item["id"]
                fused[doc_id] = fused.get(doc_id, 0.0) + source_weight / (settings.rrf_k + rank)
                documents[doc_id] = item
                provenance.setdefault(doc_id, []).extend(item.get("query_ranks", []))
        ordered = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)[:k]
        return [{**documents[doc_id], "score": fused[doc_id],
                 "matched_queries": list(dict.fromkeys(
                     item["query"] for item in provenance[doc_id] if "query" in item)),
                 "query_ranks": provenance[doc_id]} for doc_id in ordered]

    def search(self, query: str, top_k: int | None = None,
               exclude_ids: list[str] | None = None) -> list[dict[str, Any]]:
        k = max(1, min(top_k or self.top_k, len(self.corpus)))
        excluded = {str(doc_id) for doc_id in (exclude_ids or [])}
        if self.retrieval_mode == "bm25":
            return self._search_bm25(query, k, excluded)
        if self.retrieval_mode in {"dense", "dense_instruct"}:
            return self._search_dense(query, k, excluded,
                                      instruction_aware=self.retrieval_mode == "dense_instruct")
        pool_k = min(max(k, settings.hybrid_candidate_pool), len(self.corpus))
        bm25_results = self._search_bm25(query, pool_k, excluded)
        dense_results = self._search_dense(query, pool_k, excluded, instruction_aware=True)
        return self._rrf_fuse([bm25_results, dense_results], k,
                              [settings.sparse_rrf_weight, settings.dense_rrf_weight])

    def search_many(self, queries: list[str], top_k: int | None = None,
                    exclude_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Fuse independent clue searches using the configured ranking policy."""
        k = max(1, min(top_k or self.top_k, len(self.corpus)))
        candidate_k = min(max(k * 2, k), len(self.corpus))
        fused: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        query_count: dict[str, int] = {}
        documents: dict[str, dict[str, Any]] = {}
        query_ranks: dict[str, list[dict[str, Any]]] = {}
        for query in queries:
            candidates = self.search(query, top_k=candidate_k, exclude_ids=exclude_ids)
            for rank, item in enumerate(candidates, start=1):
                doc_id = item["id"]
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (settings.rrf_k + rank)
                best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
                query_count[doc_id] = query_count.get(doc_id, 0) + 1
                documents[doc_id] = item
                query_ranks.setdefault(doc_id, []).append({"query": query, "rank": rank})
        if settings.query_fusion_mode == "rrf":
            ordered = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)
        elif settings.query_fusion_mode == "best_rank":
            ordered = sorted(fused, key=lambda doc_id: (
                best_rank[doc_id], -query_count[doc_id], doc_id))
        elif settings.query_fusion_mode == "clue_coverage":
            # Retrieval provenance is only a ranking hint: favor candidates
            # surfaced by multiple independent clue queries, then preserve
            # reciprocal-rank relevance. Opened text must still verify claims.
            denominator = max(1, len(queries) - 1)
            coverage_scores = {
                doc_id: fused[doc_id] + settings.query_coverage_weight
                * (query_count[doc_id] - 1) / denominator
                for doc_id in fused
            }
            ordered = sorted(fused, key=lambda doc_id: (
                -coverage_scores[doc_id], -query_count[doc_id], best_rank[doc_id], doc_id))
        else:
            raise ValueError(
                f"Unsupported QUERY_FUSION_MODE {settings.query_fusion_mode!r}; "
                "choose 'rrf', 'best_rank', or 'clue_coverage'."
            )
        ranked = ordered[:k]
        return [{**documents[doc_id], "score": fused[doc_id],
                 "matched_queries": [item["query"] for item in query_ranks[doc_id]],
                 "query_ranks": query_ranks[doc_id]} for doc_id in ranked]

    def open_document(self, doc_id: str) -> dict[str, Any]:
        row_idx = self._lookup().get(str(doc_id))
        if row_idx is None:
            return {"error": f"Document {doc_id!r} was not found"}
        row = self.corpus[row_idx]
        return {"id": str(row["docid"]), "text": str(row["text"]), "url": str(row["url"])}

    def find_text(self, doc_id: str, phrase: str) -> dict[str, Any]:
        doc = self.open_document(doc_id)
        if "error" in doc:
            return doc
        if not phrase.strip():
            return {"id": doc_id, "matches": [], "message": "Phrase cannot be empty"}
        text = doc["text"]
        pos = text.casefold().find(phrase.casefold())
        if pos < 0:
            return {"id": doc_id, "matches": [], "message": "Phrase not found"}
        return {"id": doc_id, "matches": [text[max(0, pos - 300):pos + len(phrase) + 500]]}


def create_search_tools(corpus: Dataset, bm25_index: bm25s.BM25,
                        top_k: int | None = None) -> SearchTools:
    """Construct configured retrievers and validate dense-index corpus alignment."""
    dense_index = None
    if settings.retrieval_mode != "bm25":
        dense_index = DenseIndex(
            settings.dense_index_dir, settings.dense_model,
            expected_doc_ids=[str(value) for value in corpus["docid"]],
        )
    return SearchTools(
        corpus, bm25_index, top_k or settings.top_k,
        retrieval_mode=settings.retrieval_mode, dense_index=dense_index,
    )

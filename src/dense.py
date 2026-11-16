"""Dense embedding index build/load and query search helpers."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


INSTRUCTION_PREFIX = "Represent this research question for retrieving relevant evidence: "


class DenseIndex:
    def __init__(self, index_dir: Path, model_name: str, expected_doc_ids: list[str]):
        metadata_path = index_dir / "metadata.json"
        vectors_path = index_dir / "embeddings.npy"
        ids_path = index_dir / "docids.json"
        if not metadata_path.is_file() or not vectors_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(
                f"Dense index is incomplete in {index_dir}. Build it with `python -m src.build_dense_index`."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("model") != model_name:
            raise ValueError(
                f"Dense index uses model {self.metadata.get('model')!r}, but DENSE_MODEL is {model_name!r}. "
                "Rebuild the dense index or restore the matching setting."
            )
        self.doc_ids = [str(value) for value in json.loads(ids_path.read_text(encoding="utf-8"))]
        if self.doc_ids != [str(value) for value in expected_doc_ids]:
            raise ValueError("Dense index document order does not match the loaded corpus; rebuild the dense index.")
        self.vectors = np.load(vectors_path, mmap_mode="r")
        if (self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.doc_ids)
                or self.metadata.get("documents") != len(self.doc_ids)
                or self.metadata.get("dimension") != self.vectors.shape[1]):
            raise ValueError("Dense index vector shape does not match its document ID list.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the project requirements to use dense retrieval: pip install -r requirements.txt") from exc
        self.encoder = SentenceTransformer(model_name)
        self.rrf_k = 60

    @staticmethod
    def build(corpus: Any, index_dir: Path, model_name: str,
              dataset_name: str = "configured corpus", batch_size: int = 128,
              instruction: str = "") -> None:
        """Encode corpus rows in batches and persist normalized vectors on disk."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the project requirements to build dense retrieval: pip install -r requirements.txt") from exc
        index_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for {model_name!r}.")
        vector_path = index_dir / "embeddings.npy"
        temporary_vector_path = index_dir / "embeddings.building.npy"
        state_path = index_dir / "build_state.json"
        doc_ids = [str(value) for value in corpus["docid"]]
        doc_ids_hash = hashlib.sha256("\n".join(doc_ids).encode("utf-8")).hexdigest()
        expected_state = {"model": model_name, "documents": len(corpus),
                          "dimension": int(dimension), "docids_sha256": doc_ids_hash}
        completed = 0
        if temporary_vector_path.is_file() and state_path.is_file():
            try:
                prior_state = json.loads(state_path.read_text(encoding="utf-8"))
                if all(prior_state.get(key) == value for key, value in expected_state.items()):
                    completed = max(0, min(int(prior_state.get("completed", 0)), len(corpus)))
            except (OSError, ValueError, TypeError):
                completed = 0
        if completed:
            print(f"Resuming dense index at {completed:,}/{len(corpus):,} documents.", flush=True)
        vectors = np.lib.format.open_memmap(
            temporary_vector_path, mode="r+" if completed else "w+",
            dtype=np.float32, shape=(len(corpus), dimension)
        )
        texts = corpus["text"]
        for start in range(completed, len(corpus), batch_size):
            batch = [str(text) for text in texts[start:start + batch_size]]
            if instruction:
                batch = [f"{instruction}{text}" for text in batch]
            encoded = model.encode(
                batch, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            vectors[start:start + len(batch)] = encoded.astype(np.float32, copy=False)
            completed = start + len(batch)
            vectors.flush()
            checkpoint = {**expected_state, "completed": completed}
            checkpoint_tmp = state_path.with_suffix(".tmp")
            checkpoint_tmp.write_text(json.dumps(checkpoint), encoding="utf-8")
            checkpoint_tmp.replace(state_path)
            print(f"Encoded {min(start + len(batch), len(corpus)):,}/{len(corpus):,} documents", flush=True)
        vectors.flush()
        del vectors
        temporary_vector_path.replace(vector_path)
        (index_dir / "docids.json").write_text(json.dumps(doc_ids), encoding="utf-8")
        metadata = {
            "dataset": dataset_name,
            "documents": len(doc_ids),
            "model": model_name,
            "dimension": int(dimension),
            "normalized": True,
            "document_instruction": instruction,
        }
        (index_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state_path.unlink(missing_ok=True)

    def search(self, query: str, top_k: int, instruction_aware: bool = False,
               exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
        prompt = f"{INSTRUCTION_PREFIX}{query}" if instruction_aware else query
        vector = self.encoder.encode(
            [prompt], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        scores = self.vectors @ vector
        k = max(1, min(top_k, len(self.doc_ids)))
        # Argpartition limits work to the requested candidates; a final sort gives
        # deterministic score order for the selected set.
        candidate_count = min(len(self.doc_ids), k + len(exclude_ids or set()))
        candidates = np.argpartition(scores, len(scores) - candidate_count)[-candidate_count:]
        ranked = sorted(candidates.tolist(), key=lambda idx: float(scores[idx]), reverse=True)
        excluded = exclude_ids or set()
        results = []
        for rank, index in enumerate(ranked, start=1):
            doc_id = self.doc_ids[index]
            if doc_id in excluded:
                continue
            results.append({"id": doc_id, "score": float(scores[index]), "rank": rank})
            if len(results) >= k:
                break
        return results

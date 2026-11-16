from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    validation_dataset: str = os.getenv("VALIDATION_DATASET", "HongruCai/COMP5423-BrowseComp-Plus-Val")
    test_dataset: str = os.getenv("TEST_DATASET", "HongruCai/COMP5423-BrowseComp-Plus-Test")
    corpus_dataset: str = os.getenv("CORPUS_DATASET", "Tevatron/browsecomp-plus-corpus")
    data_cache_dir: Path = configured_path("DATA_CACHE_DIR", PROJECT_ROOT / "data" / "hf_cache")
    index_dir: Path = configured_path("INDEX_DIR", PROJECT_ROOT / "data" / "index")
    dense_index_dir: Path = configured_path("DENSE_INDEX_DIR", PROJECT_ROOT / "data" / "dense_index")
    model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    max_agent_steps: int = max(1, int(os.getenv("MAX_AGENT_STEPS", "8")))
    max_action_retries: int = max(0, int(os.getenv("MAX_ACTION_RETRIES", "2")))
    max_completion_tokens: int = max(1024, int(os.getenv("MAX_COMPLETION_TOKENS", "8192")))
    max_search_calls: int = int(os.getenv("MAX_SEARCH_CALLS", "3"))
    max_additional_queries: int = min(5, max(0, int(os.getenv("MAX_ADDITIONAL_QUERIES", "2"))))
    top_k: int = int(os.getenv("TOP_K", "8"))
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "bm25").strip().lower()
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    rrf_k: int = max(1, int(os.getenv("RRF_K", "60")))
    query_fusion_mode: str = os.getenv("QUERY_FUSION_MODE", "best_rank").strip().lower()
    query_coverage_weight: float = max(0.0, float(os.getenv("QUERY_COVERAGE_WEIGHT", "0.02")))
    search_snippet_chars: int = min(5000, max(200, int(os.getenv("SEARCH_SNIPPET_CHARS", "700"))))
    hybrid_candidate_pool: int = max(1, int(os.getenv("HYBRID_CANDIDATE_POOL", "100")))
    sparse_rrf_weight: float = max(0.0, float(os.getenv("SPARSE_RRF_WEIGHT", "1.0")))
    dense_rrf_weight: float = max(0.0, float(os.getenv("DENSE_RRF_WEIGHT", "1.0")))
    min_identity_sources: int = min(4, max(1, int(os.getenv("MIN_IDENTITY_SOURCES", "2"))))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "18000"))
    auto_open_first_candidate: bool = os.getenv("AUTO_OPEN_FIRST_CANDIDATE", "true").strip().lower() in {"1", "true", "yes", "on"}
    auto_open_candidate_count: int = min(3, max(1, int(os.getenv("AUTO_OPEN_CANDIDATE_COUNT", "1"))))
    explicit_candidate_state: bool = os.getenv("EXPLICIT_CANDIDATE_STATE", "false").strip().lower() in {"1", "true", "yes", "on"}
    structured_evidence_ledger: bool = os.getenv("STRUCTURED_EVIDENCE_LEDGER", "false").strip().lower() in {"1", "true", "yes", "on"}
    enable_skills: bool = os.getenv("ENABLE_SKILLS", "true").strip().lower() in {"1", "true", "yes", "on"}


settings = Settings()

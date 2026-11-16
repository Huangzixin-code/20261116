# COMP5423 Skill-Augmented Search Agent

This is the fresh project workspace for the COMP5423 Fall 2026 group project.

## 1. Prepare Python environment (PowerShell)

```powershell
cd D:\COMP5423
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `DEEPSEEK_API_KEY` in `.env`. Keep the real key private and do not commit `.env`.

## 2. Download and build the corpus index

The validation dataset is loaded with `datasets.load_dataset("HongruCai/COMP5423-BrowseComp-Plus-Val", split="validation")`. The fixed document corpus is loaded from `Tevatron/browsecomp-plus-corpus`, expected columns `docid`, `text`, and `url`, 100,195 rows. Both Hugging Face cache and index default to `D:\COMP5423\data`.

```powershell
& .\.venv\Scripts\python.exe -m src.build_index
```

The first run downloads the corpus. It tokenizes the complete corpus and persists a BM25S index under `INDEX_DIR`; this can use substantial RAM and disk space. Later app runs load that saved index instead of rebuilding it. If a valid Hugging Face cache already exists elsewhere, set `DATA_CACHE_DIR` to its exact path in `.env` to reuse it.

## 3. Build optional dense retrieval index

Install the added embedding dependency and build the vectors once:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m src.build_dense_index --batch-size 128
```

The default encoder is `BAAI/bge-small-en-v1.5`. It downloads model weights to the Hugging Face cache and writes a normalized float32 vector per corpus document to `DENSE_INDEX_DIR` (roughly 150 MB for 100k documents, plus the model cache). Building can take a while on CPU. The builder reports progress and checkpoints each batch so it can resume after interruption. It stores document IDs; runtime checks that their order matches the loaded corpus.

Choose a retrieval mode in the local `.env`:

```dotenv
RETRIEVAL_MODE=bm25
# or: dense, dense_instruct, hybrid
```

`dense` uses vector similarity without a query instruction. `dense_instruct` adds a retrieval instruction to the question before encoding. `hybrid` fuses BM25 and instruction-aware dense rankings with reciprocal-rank fusion; it retrieves up to `HYBRID_CANDIDATE_POOL` candidates from each retriever before returning `TOP_K`. `SPARSE_RRF_WEIGHT` and `DENSE_RRF_WEIGHT` adjust their contributions (both default to 1.0). Keep `bm25` for the existing baseline. Rebuild the dense index only when changing `DENSE_MODEL` or the corpus.

When a search action contains multiple clue queries, `QUERY_FUSION_MODE=best_rank` ranks a document by its best rank under any query, breaking ties by the number of queries that found it. This is the current default based on a 30-question exploratory pilot. Set `QUERY_FUSION_MODE=rrf` to restore reciprocal-rank fusion for comparison. This option affects multi-query BM25 fusion; hybrid retrieval continues to use its configured `RRF_K` and retriever weights.

`SEARCH_SNIPPET_CHARS` controls how many characters of each search result are shown to the agent. The default is 700; larger excerpts can make prompts longer and have not improved end-to-end accuracy in the current pilot.

`MAX_ADDITIONAL_QUERIES` sets how many independent clue queries a search action may include beyond its primary query. The default is 2; allowing 3 did not improve the paired validation pilot.

`AUTO_OPEN_FIRST_CANDIDATE=true` opens the highest-ranked result after the first successful search so the agent sees its full text before deciding on more searches. `AUTO_OPEN_CANDIDATE_COUNT` controls how many top results to open (default `1`); opening two candidates did not improve the same-range pilot. The current full validation favored opening one over the off baseline; set the first option to `false` for an ablation.

For multi-hop identity questions, `MIN_IDENTITY_SOURCES` controls the minimum number of distinct opened sources the final answer must cite (default `2`). A source can support multiple clues only if its text explicitly contains them; setting this to `1` is useful for a controlled evidence-gate ablation.

Once the index exists, compare retrieval-only candidate coverage on the same labeled questions without making DeepSeek calls:

```powershell
& .\.venv\Scripts\python.exe -m src.evaluate_retrieval --split validation --limit 10 --top-k 8
```

This reports literal gold-answer coverage and reciprocal rank. These are post-hoc retrieval diagnostics, not end-to-end answer accuracy; use the end-to-end evaluator for benchmark results.

## 4. Launch the UI

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

## 5. Run the agent in Jupyter Notebook

`COMP5423_SearchAgent.ipynb` is the consolidated project report and runnable demo. It includes the assignment requirement mapping, system design, labeled workflow, implementation notes, actual validation and retrieval results, experiment analysis, UI design, limitations, and an interactive question-answering section. It imports the same implementation as the Streamlit UI and reads the existing `.env`, Hugging Face cache, and index; it does not duplicate the agent implementation. The actual Streamlit screenshot is still to be captured into `report\streamlit_ui_screenshot.png`; the notebook displays it automatically when present.

Install the optional notebook dependencies once, then launch the notebook:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\run_notebook.ps1
```

In VS Code, open the notebook and select `D:\COMP5423\.venv\Scripts\python.exe` as the kernel. Run the numbered cells in order. Each question calls the configured DeepSeek API. The optional final cell appends a run record to `results\notebook_runs.jsonl`.

## 6. Evaluate and analyze validation

```powershell
& .\.venv\Scripts\python.exe -m src.evaluate --split validation --limit 100
& .\.venv\Scripts\python.exe -m src.analyze_results results\validation_predictions.jsonl
```

For the course's blind final test, the project uses `TEST_DATASET=HongruCai/COMP5423-BrowseComp-Plus-Test` separately from the validation labels. First download/cache it with the notebook's test-data preparation cell (or `src.data.load_test_dataset(local_files_only=False)`), then freeze the best configuration and run `& .\.venv\Scripts\python.exe -m src.evaluate --split test --output results\final_test_predictions.jsonl`. The test loader accepts a `test` split or a single-split repository, requires only `id` and `question`, and never uses gold answers. The prediction JSONL contains `id`, `question`, generated `answer`, trajectory, and usage metrics; keep the `.meta.json` sidecar. Do not run `src.analyze_results` on the blind test set. Preserve the full trajectory file and create a minimal submission file only if the course specifies its required schema.

Use `--offset` to evaluate a non-overlapping range (zero-based), for example `--offset 5 --limit 5` evaluates rows 6–10. Each run writes a `.meta.json` sidecar with the intended range and split size, so analysis reports accuracy and prediction coverage even if the run stops before its first prediction. For legacy prediction files without metadata, analysis scores the unique known IDs present. The prediction JSONL embeds each generated trajectory and token/tool counts. The answer field contains the system prediction, not the gold label.

The agent uses DeepSeek function calls for typed actions and validates each call before execution. Malformed function arguments and final-turn tool calls get up to `MAX_ACTION_RETRIES` corrections without consuming a tool step. If a state-invalid search action is requested (for example, a repeated query, a premature next search, or an exhausted search budget), or the model tries to open an already inspected document, the agent opens the highest-scoring retrieved document that has not been inspected and records the reason in its trajectory. Retrieval scores are retained in agent state for this recovery. Context trimming preserves complete assistant function-call and tool-result pairs to keep the conversation valid for the API. Truncated outputs are identified from the API finish reason; `MAX_COMPLETION_TOKENS` defaults to `8192`, and the limit increases on truncated retries. The last configured agent step is reserved for an answer. Prediction rows record model calls, correction count, and stop reason for later analysis. `src.analyze_results --details` reports per-question correctness, citations, opened documents, and requested versus executed searches. Set `MAX_ACTION_RETRIES` in `.env` to change the retry cap (default: `2`).

## Project layout

- `src/data.py`: Hugging Face loading and schema validation.
- `src/build_index.py`: one-time BM25S index construction and persistence.
- `src/build_dense_index.py`, `src/dense.py`: optional dense index construction and search.
- `src/retrieval.py`: configurable BM25, dense, instruction-aware dense, or hybrid retrieval plus open-document and phrase-finding tools.
- `src/agent.py`: DeepSeek-driven iterative loop, structured state, skills, trajectory, token counts.
- `src/app.py`: interactive UI.
- `skills/`: two authored runtime skills.
- `report/`: report draft and actual-experiment table.

## Current implementation scope

The retained runtime configuration is sparse BM25 with best-rank multi-query fusion, top-k 8, and 700-character search snippets. Dense and instruction-aware dense modes require the optional dense index to be built before selection. The report and experiment tables contain completed ablations and validation results; a final UI screenshot and report formatting remain to be completed before submission. Never put fabricated values in the report.

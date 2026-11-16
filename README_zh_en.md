# COMP5423 Skill-Augmented Search Agent
# COMP5423 技能增强型搜索智能体

**Course / 课程:** COMP5423 Natural Language Processing / 自然语言处理  
**Project / 项目:** Skill-Augmented Search Agent on BrowseComp-Plus / 基于 BrowseComp-Plus 的技能增强型搜索智能体

This repository implements an iterative research agent that searches a fixed document corpus, opens and checks evidence, then returns a concise answer with source IDs. The implementation is shared by the command-line evaluator, Jupyter Notebook, and Streamlit interface.

本项目实现了一个迭代式研究智能体：它在固定文档语料中检索资料、打开并核验证据，最后生成简洁答案及来源 ID。命令行评测器、Jupyter Notebook 和 Streamlit 界面共用同一套实现。

## 1. System Overview / 系统概览

- **Dataset / 数据集:** 100,195-document BrowseComp-Plus corpus; 100-question course validation split. The separate course test split contains 100 questions.  
  **数据集：** BrowseComp-Plus 固定语料包含 100,195 篇文档；课程 validation 集包含 100 道题。独立的课程 test 集包含 100 道题。
- **Retriever / 检索器:** BM25S sparse retrieval with best-rank fusion for independent clue queries. An optional BGE dense index and hybrid retriever are also available.  
  **检索器：** 使用 BM25S 稀疏检索，并按 best-rank 方式融合不同线索的查询。项目还支持可选的 BGE 稠密索引和混合检索。
- **Agent / 智能体:** DeepSeek V4 Flash function calling selects among `search`, `open`, `find`, and `answer`.  
  **智能体：** DeepSeek V4 Flash 通过 function calling 在 `search`、`open`、`find` 和 `answer` 工具间选择。
- **Skills / 技能:** Runtime guidance in `skills/` supports multi-hop identity research and search recovery.  
  **技能：** `skills/` 中的运行时指南支持多跳身份检索和搜索过程恢复。
- **Interfaces / 使用界面:** Jupyter Notebook, command-line evaluation, and Streamlit.  
  **使用界面：** Jupyter Notebook、命令行评测和 Streamlit。

The agent uses only the supplied fixed corpus; it does not perform live web search. Citations are checked against documents opened during the run.

智能体只使用项目提供的固定语料，不进行实时网页搜索。程序会检查引用的文档是否在本轮运行中被打开。

## 2. Environment Setup / 环境配置

Run these commands in PowerShell from the project root:

在项目根目录通过 PowerShell 运行：

```powershell
cd D:\COMP5423
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `DEEPSEEK_API_KEY` in the local `.env` file. Keep the key private and never commit it or paste it into a report.

在本地 `.env` 文件中填写 `DEEPSEEK_API_KEY`。请妥善保管密钥，不要提交到版本库，也不要粘贴到报告中。

Important dataset settings in `.env`:

`.env` 中的重要数据集配置：

```dotenv
VALIDATION_DATASET=HongruCai/COMP5423-BrowseComp-Plus-Val
TEST_DATASET=HongruCai/COMP5423-BrowseComp-Plus-Test
CORPUS_DATASET=Tevatron/browsecomp-plus-corpus
```

The test dataset is independent of validation and should be cached before running the final evaluation. The loader accepts a `test` split or a single-split repository and requires `id` and `question` fields.

测试集与 validation 集独立，最终评测前需先缓存。加载器支持名为 `test` 的 split 或仅包含一个 split 的数据集仓库，并要求数据包含 `id` 和 `question` 字段。

## 3. Download Data and Build the Index / 下载数据并构建索引

The first setup requires network access to Hugging Face. The project stores its cache under `D:\COMP5423\data\hf_cache` by default. Download the corpus and build the BM25 index with:

首次准备需要访问 Hugging Face。项目默认将缓存保存在 `D:\COMP5423\data\hf_cache`。使用以下命令下载语料并构建 BM25 索引：

```powershell
& .\.venv\Scripts\python.exe -m src.build_index
```

Later evaluation runs use the local dataset and index caches. To build the optional BGE dense index:

后续评测会使用本地数据集和索引缓存。若要构建可选的 BGE 稠密索引：

```powershell
& .\.venv\Scripts\python.exe -m src.build_dense_index --batch-size 128
```

Dense and hybrid retrieval require the dense index. BM25 is the retained default and does not require it.

稠密检索和混合检索需要先有稠密索引。当前保留的默认方案为 BM25，不依赖稠密索引。

## 4. Run the System / 运行系统

### Jupyter Notebook / Jupyter Notebook

Open `COMP5423_SearchAgent.ipynb` in VS Code and select `D:\COMP5423\.venv\Scripts\python.exe` as the kernel. The Notebook includes the system overview, local validation workflow, interactive question answering, and a final test-set workflow.

在 VS Code 中打开 `COMP5423_SearchAgent.ipynb`，并选择 `D:\COMP5423\.venv\Scripts\python.exe` 作为 Kernel。Notebook 包含系统说明、本地 validation 流程、交互式问答和最终测试集流程。

For the final test workflow, restart the kernel and run the notebook setup cells from the beginning if Hugging Face offline mode was previously enabled. First run the test-data cache cell. After confirming the best configuration and API key, set `RUN_FINAL_TEST = True` in the final test cell. This runs all 100 test questions using the retained validation configuration.

运行最终测试集流程时，如果之前启用了 Hugging Face 离线模式，请先重启 Kernel 并从头运行 Notebook 初始化单元。先运行测试集缓存单元；确认最佳配置和 API 密钥无误后，将最终测试单元中的 `RUN_FINAL_TEST = True`，即可用当前保留的 validation 配置运行全部 100 道测试题。

### Command Line / 命令行

Run all 100 validation questions and analyze the labeled validation results:

运行全部 100 道 validation 题，并分析带标签的 validation 结果：

```powershell
& .\.venv\Scripts\python.exe -m src.evaluate --split validation --offset 0 --limit 100 --output results\validation_predictions.jsonl
& .\.venv\Scripts\python.exe -m src.analyze_results results\validation_predictions.jsonl
```

After the test split has been cached, generate final test predictions with:

测试集缓存完成后，使用以下命令生成最终测试预测：

```powershell
& .\.venv\Scripts\python.exe -m src.evaluate --split test --offset 0 --limit 100 --output results\final_test_predictions.jsonl
```

The evaluator writes a detailed trajectory file and a `.meta.json` sidecar containing the run range and configuration. The Notebook also creates `results\final_test_submission.jsonl` with only `id` and generated `answer` fields.

评测器会生成包含完整运行轨迹的预测文件，以及记录评测范围和配置的 `.meta.json` 侧车文件。Notebook 还会生成仅包含 `id` 和模型预测 `answer` 的 `results\final_test_submission.jsonl`。

### Streamlit UI / Streamlit 界面

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Enter a research question, click **Search**, then inspect the answer, collected evidence, and agent trajectory.

输入研究问题后点击 **Search**，即可查看答案、收集到的证据和智能体运行轨迹。

## 5. Evaluation Protocol and Results / 评测流程与结果

The current best validation run is `results/optimization_min_sources1_strict_validation100.jsonl`: normalized exact match (EM) **33/100 (33%)** on the full validation set. Its configuration uses BM25, best-rank query fusion, top-k 8, 700-character search snippets, up to three search actions, up to eight agent steps, automatic opening of the first candidate, and a one-source minimum for multi-hop identity answers.

当前最佳 validation 结果保存在 `results/optimization_min_sources1_strict_validation100.jsonl`：完整 validation 集上的 normalized exact match（EM）为 **33/100（33%）**。该配置使用 BM25、best-rank 查询融合、top-k 8、700 字符搜索片段、最多三次搜索、最多八步、自动打开首个候选，并要求多跳身份答案至少引用一个来源。

The validation score is not the test score. The test set contains only `id` and `question`; it intentionally provides no reference answers. Therefore, the project can generate and validate 100 predictions locally, but cannot calculate test EM locally. The test set is for final evaluation only and must not be used to tune parameters.

Validation 分数不等于测试分数。测试集只包含 `id` 和 `question`，不会提供参考答案。因此，项目可以在本地生成并校验 100 条预测，但不能在本地计算测试 EM。测试集只用于最终评测，不应被用于调参。

The most recent final test artifacts are:

最近一次最终测试文件：

- `results/final_test_predictions.jsonl` — full predictions, trajectories, and usage metrics. / 完整预测、轨迹及用量统计。
- `results/final_test_submission.jsonl` — one `id` and generated `answer` per line. / 每行一个 `id` 和模型生成的 `answer`。
- `results/final_test_predictions.jsonl.meta.json` — dataset range and configuration. / 数据集范围与运行配置。

Check the course submission example for the required filename and JSONL field names before uploading.

提交前请按课程给出的提交示例核对文件名和 JSONL 字段名。

## 6. Project Structure / 项目结构

| Path / 路径 | Description / 说明 |
|---|---|
| `src/data.py` | Hugging Face loaders and dataset schema checks / Hugging Face 数据加载和字段校验 |
| `src/build_index.py` | BM25 index builder / BM25 索引构建 |
| `src/retrieval.py`, `src/dense.py` | Sparse, dense, and hybrid retrieval / 稀疏、稠密和混合检索 |
| `src/agent.py`, `src/skills.py` | Agent loop, action validation, evidence tracking, runtime skills / 智能体循环、动作校验、证据跟踪和运行时技能 |
| `src/evaluate.py` | Generate validation or test predictions / 生成 validation 或 test 预测 |
| `src/analyze_results.py` | Score labeled validation results / 评测带标签的 validation 结果 |
| `src/app.py` | Streamlit UI / Streamlit 界面 |
| `skills/` | Runtime skill instructions / 运行时技能说明 |
| `COMP5423_SearchAgent.ipynb` | Main project and final test notebook / 项目主 Notebook 和最终测试流程 |
| `COMP5423_SearchAgent_Optimization_v3.ipynb` | Optimization experiments / 优化实验记录 |

## 7. Security and Reproducibility / 安全与复现

- Never place API keys in source files, notebooks, JSONL predictions, screenshots, or reports. / 不要把 API 密钥写入源代码、Notebook、JSONL 预测、截图或报告。
- Preserve the test prediction file and its metadata together. / 请同时保留测试预测文件及其元数据。
- Do not run the validation analyzer on the blind test split; it has no local gold labels. / 盲测集没有本地参考答案，不要对 test split 运行 validation 分析器。
- Keep validation and test results separate. / 请将 validation 和 test 结果分开保存。

## Citation / 引用

The fixed corpus and question design are based on BrowseComp-Plus. Please cite the original benchmark and the course-provided Hugging Face datasets as required by the assignment.

固定语料和问题设计基于 BrowseComp-Plus。请根据课程要求引用原始基准及课程提供的 Hugging Face 数据集。

# RAG Over Codebase

This repo is implementing the capstone project phase by phase in Python.

## Phase 1: AST-Aware Ingestion

Install dependencies:

```bash
pip install -r requirements.txt
```

Run ingestion:

```bash
python -m packages.ingestion.cli --repo-path ./sample-repo --repo-name sample-repo --out data/chunks/sample.jsonl
```

Run tests:

```bash
python -m unittest discover -s packages/tests
```

## Phase 2: Hybrid Indexing

Phase 2 reads Phase 1 JSONL chunks, creates cached summaries, builds deterministic dense vectors, and writes a weighted sparse BM25 index.

The current implementation is local-first:

- Dense search: local JSON vector store, shaped so it can be replaced by Qdrant.
- Sparse search: SQLite FTS5 BM25 with weights `symbol_name=4`, `summary=2`, `code_body=1`, shaped so it can be replaced by Tantivy.
- Summaries: heuristic provider with cache by `content_hash`, shaped so it can be replaced by Gemini/Claude.

Build indexes:

```bash
python -m packages.indexing.cli index --chunks data/chunks/sample.jsonl
```

Search indexes:

```bash
python -m packages.indexing.cli search --q "verify token" --top-k 5
```

Use real local retrieval backends:

```bash
python -m packages.indexing.cli index \
  --chunks data/chunks/sample.jsonl \
  --embedding-provider fastembed \
  --embedding-model jinaai/jina-embeddings-v2-base-code \
  --dense-backend qdrant \
  --dense-index data/qdrant \
  --sparse-backend tantivy \
  --sparse-index data/tantivy

python -m packages.indexing.cli search \
  --q "verify token" \
  --embedding-provider fastembed \
  --embedding-model jinaai/jina-embeddings-v2-base-code \
  --dense-backend qdrant \
  --dense-index data/qdrant \
  --sparse-backend tantivy \
  --sparse-index data/tantivy
```

For a faster local smoke test, use the smaller real embedding model:

```bash
python -m packages.indexing.cli index \
  --chunks data/chunks/sample.jsonl \
  --embedding-provider fastembed \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --dense-backend qdrant \
  --dense-index data/qdrant-smoke \
  --sparse-backend tantivy \
  --sparse-index data/tantivy-smoke
```

## Phase 3: Symbol Graph

Phase 3 builds a local symbol graph from Phase 1 chunks plus Tree-sitter AST analysis.

The current implementation is local-first:

- Graph store: SQLite nodes/edges, shaped so it can be replaced by KuzuDB or Neo4j.
- Nodes: `Repo`, `File`, `Function`, `Class`, plus `External` for unresolved imports/calls.
- Edges: `CONTAINS`, `IMPORTS`, `CALLS`.
- Resolution: exact/suffix symbol matching, with same-file matches preferred.

Build graph:

```bash
python -m packages.graph.cli build --repo-path packages/tests/fixtures/sample_repo --repo-name sample_repo --chunks data/chunks/sample.jsonl
```

Build graph in Neo4j:

```bash
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="password"

python -m packages.graph.cli build \
  --backend neo4j \
  --repo-path packages/tests/fixtures/sample_repo \
  --repo-name sample_repo \
  --chunks data/chunks/sample.jsonl
```

Expand context around a chunk:

```bash
python -m packages.graph.cli expand --chunk-id "sample_repo:src/auth/service.py:AuthService.verify_token:9:11"
```

## Phase 4: Agentic Search With LangChain

Phase 4 keeps the custom hybrid retrieval stack and uses LangChain for the final
LLM orchestration:

```text
Qdrant dense + Tantivy BM25 -> reciprocal-rank-fusion -> graph expansion -> LangChain -> LM Studio
```

Ask with LM Studio running at `http://localhost:1234/v1`:

```bash
python -m packages.agent.cli ask --q "Neo4j graph store dùng ở đâu?"
```

Print only the final answer:

```bash
python -m packages.agent.cli ask --q "solve_dependencies trong FastAPI lam gi?" --answer-only
```

Test retrieval and context without calling LM Studio:

```bash
python -m packages.agent.cli ask --q "Neo4j graph store dùng ở đâu?" --no-synth --include-debug
```
## RepoBench-R Retrieval Eval

Prepare a small RepoBench-R split from HuggingFace parquet data:

```bash
python -m packages.evals.repobench_r_cli prepare \
  --hf-config python_cff \
  --split test_easy \
  --limit 1000 \
  --chunks-out data/evals/repobench_r_chunks.jsonl \
  --queries-out data/evals/repobench_r_queries.jsonl
```

Index the prepared candidate chunks:

```bash
python -m packages.indexing.cli index \
  --chunks data/evals/repobench_r_chunks.jsonl \
  --indexed-chunks data/evals/repobench_r_indexed.jsonl \
  --dense-index data/evals/qdrant_repobench_r \
  --sparse-index data/evals/tantivy_repobench_r \
  --summary-cache data/evals/repobench_r_summary.json \
  --embedding-provider fastembed \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --embedding-cache-dir data/models/fastembed \
  --dense-backend qdrant \
  --sparse-backend tantivy \
  --qdrant-collection repobench_r_chunks
```

Run retrieval metrics:

```bash
python -m packages.evals.repobench_r_cli run \
  --queries data/evals/repobench_r_queries.jsonl \
  --dense-index data/evals/qdrant_repobench_r \
  --sparse-index data/evals/tantivy_repobench_r \
  --embedding-provider fastembed \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --embedding-cache-dir data/models/fastembed \
  --dense-backend qdrant \
  --sparse-backend tantivy \
  --qdrant-collection repobench_r_chunks \
  --mode hybrid \
  --scope sample \
  --candidate-top-k 50 \
  --rerank-method bge-m3 \
  --bge-reranker-model BAAI/bge-reranker-v2-m3 \
  --bge-reranker-cache-dir data/models/bge-reranker \
  --bge-reranker-batch-size 8 \
  --top-k 10 \
  --retrieve-top-k 200 \
  --fused-top-k 200 \
  --report-out data/evals/repobench_r_report.json
```

## Phase 5: Incremental Sync And Eval Gates

List files changed between two git refs:

```bash
python -m packages.sync.cli changed-files \
  --repo-path data/repos/fastapi \
  --base HEAD~1 \
  --head HEAD \
  --json
```

Run incremental ingestion state diff:

```bash
python -m packages.sync.cli incremental \
  --repo-path data/repos/fastapi \
  --repo-name fastapi \
  --state data/sync/fastapi_state.json \
  --base HEAD~1 \
  --head HEAD \
  --changed-chunks-out data/sync/fastapi_changed_chunks.jsonl \
  --full-chunks-out data/sync/fastapi_full_chunks.jsonl \
  --report-out data/sync/fastapi_incremental_report.json
```

Compare two eval reports and fail if selected metrics drop too much:

```bash
python -m packages.evals.cli compare \
  --baseline data/evals/repobench_r_baseline_report.json \
  --current data/evals/repobench_r_report.json \
  --metric mrr_at_10 \
  --metric ndcg_at_10 \
  --metric recall_at_10 \
  --max-drop 0.02 \
  --report-out data/evals/repobench_r_compare.json
```

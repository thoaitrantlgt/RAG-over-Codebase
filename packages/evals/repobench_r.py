import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from packages.agent.retrieval import reciprocal_rank_fusion, retrieve
from packages.agent.config import AgentConfig
from packages.indexing.io import read_jsonl

from .metrics import RankingMetrics, average_metrics, evaluate_ranking


@dataclass(frozen=True)
class RepoBenchQuery:
    query_id: str
    query: str
    positive_chunk_ids: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepoBenchPrepResult:
    sample_count: int
    chunk_count: int
    query_count: int
    chunks_path: Path
    queries_path: Path


@dataclass(frozen=True)
class RepoBenchEvalResult:
    sample_count: int
    mode: str
    metrics: RankingMetrics
    per_query: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mode": self.mode,
            "metrics": self.metrics.to_dict(),
            "per_query": self.per_query,
        }


def load_records(
    *,
    input_path: str | Path | None = None,
    hf_config: str | None = None,
    split: str = "test",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if input_path is not None:
        records = read_json_or_jsonl(input_path)
    else:
        records = load_huggingface_records(hf_config=hf_config, split=split)

    if limit is not None:
        return records[:limit]
    return records


def prepare_repobench_records(
    records: Iterable[dict[str, Any]],
    *,
    chunks_path: str | Path,
    queries_path: str | Path,
    repo_name: str = "repobench-r",
    max_query_chars: int = 4000,
) -> RepoBenchPrepResult:
    chunks: list[dict[str, Any]] = []
    queries: list[RepoBenchQuery] = []
    sample_count = 0

    for sample_index, record in enumerate(records):
        candidates = extract_candidates(record)
        gold_indices = extract_gold_indices(record, candidates)
        if not candidates or not gold_indices:
            continue

        query_id = str(record.get("id") or record.get("task_id") or f"sample-{sample_index}")
        query = build_query(record, max_chars=max_query_chars)
        if not query.strip():
            continue

        positive_chunk_ids: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            chunk_id = make_chunk_id(repo_name, query_id, candidate_index)
            code_body = candidate["code_body"]
            path = candidate.get("path") or f"{query_id}/candidate_{candidate_index}.py"
            chunk = {
                "repo": repo_name,
                "path": path,
                "language": infer_language(path),
                "start_line": int(candidate.get("start_line") or 1),
                "end_line": int(candidate.get("end_line") or line_count(code_body)),
                "symbol_name": candidate.get("symbol_name") or f"candidate_{candidate_index}",
                "symbol_kind": "function",
                "code_body": code_body,
                "chunk_id": chunk_id,
                "content_hash": sha256(code_body.encode("utf-8")).hexdigest(),
            }
            chunks.append(chunk)
            if candidate_index in gold_indices:
                positive_chunk_ids.append(chunk_id)

        if positive_chunk_ids:
            queries.append(
                RepoBenchQuery(
                    query_id=query_id,
                    query=query,
                    positive_chunk_ids=positive_chunk_ids,
                    metadata={
                        "source_index": sample_index,
                        "repo": record.get("repo") or record.get("repo_name"),
                        "path": record.get("path") or record.get("file_path"),
                        "gold_indices": sorted(gold_indices),
                    },
                )
            )
            sample_count += 1

    write_jsonl(chunks_path, chunks)
    write_jsonl(queries_path, [query.to_dict() for query in queries])
    return RepoBenchPrepResult(
        sample_count=sample_count,
        chunk_count=len(chunks),
        query_count=len(queries),
        chunks_path=Path(chunks_path).resolve(),
        queries_path=Path(queries_path).resolve(),
    )


def run_repobench_eval(
    *,
    queries_path: str | Path,
    config: AgentConfig,
    mode: str = "hybrid",
    top_k: int = 10,
) -> RepoBenchEvalResult:
    queries = [RepoBenchQuery(**record) for record in read_jsonl(queries_path)]
    per_query: list[dict[str, Any]] = []
    metrics: list[RankingMetrics] = []

    for query in queries:
        dense, sparse = retrieve(query.query, config)
        if mode == "dense":
            ranked = dense
        elif mode == "sparse":
            ranked = sparse
        elif mode == "hybrid":
            ranked = reciprocal_rank_fusion([dense, sparse], limit=top_k, k=config.rrf_k)
        else:
            raise ValueError(f"Unsupported eval mode: {mode}")

        ranked_ids = [result.chunk_id for result in ranked[:top_k]]
        positive_ids = set(query.positive_chunk_ids)
        query_metrics = evaluate_ranking(ranked_ids, positive_ids)
        metrics.append(query_metrics)
        per_query.append(
            {
                "query_id": query.query_id,
                "metrics": query_metrics.to_dict(),
                "positive_chunk_ids": query.positive_chunk_ids,
                "ranked_chunk_ids": ranked_ids,
                "metadata": query.metadata,
            }
        )

    return RepoBenchEvalResult(
        sample_count=len(queries),
        mode=mode,
        metrics=average_metrics(metrics),
        per_query=per_query,
    )


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return read_jsonl(input_path)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON shape in {input_path}")


def load_huggingface_records(*, hf_config: str | None, split: str) -> list[dict[str, Any]]:
    if not hf_config:
        raise ValueError("HuggingFace RepoBench-R loading requires --hf-config")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading RepoBench-R from HuggingFace requires datasets. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    data_files = {
        split: (
            "hf://datasets/tianyang/repobench-r@refs/convert/parquet/"
            f"{hf_config}/{split}/*.parquet"
        )
    }
    dataset = load_dataset("parquet", data_files=data_files, split=split)
    return [dict(record) for record in dataset]


def extract_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    for field in (
        "candidates",
        "candidate_snippets",
        "retrieval_snippets",
        "context",
        "cross_file_context",
        "documents",
        "snippets",
    ):
        value = maybe_json(record.get(field))
        if isinstance(value, list):
            candidates = [normalize_candidate(item, index) for index, item in enumerate(value)]
            return [candidate for candidate in candidates if candidate["code_body"].strip()]
    return []


def normalize_candidate(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {"code_body": item, "symbol_name": f"candidate_{index}"}
    if not isinstance(item, dict):
        return {"code_body": str(item), "symbol_name": f"candidate_{index}"}

    code = first_present(
        item,
        "code",
        "code_body",
        "snippet",
        "content",
        "text",
        "body",
        "definition",
    )
    path = first_present(item, "path", "file_path", "filename", "relative_path")
    return {
        "code_body": str(code or ""),
        "path": path,
        "start_line": item.get("start_line") or item.get("start"),
        "end_line": item.get("end_line") or item.get("end"),
        "symbol_name": item.get("symbol_name") or item.get("name") or f"candidate_{index}",
    }


def extract_gold_indices(record: dict[str, Any], candidates: list[dict[str, Any]]) -> set[int]:
    for field in (
        "gold_snippet_index",
        "gold_index",
        "answer_index",
        "label",
        "labels",
        "positive_index",
        "target_index",
    ):
        value = maybe_json(record.get(field))
        if field == "labels" and is_binary_label_list(value, len(candidates)):
            return {index for index, label in enumerate(value) if label}
        indices = coerce_indices(value)
        if indices:
            return {index for index in indices if 0 <= index < len(candidates)}

    gold_snippet = record.get("gold_snippet") or record.get("answer") or record.get("target")
    if isinstance(gold_snippet, str):
        normalized = normalize_text(gold_snippet)
        return {
            index
            for index, candidate in enumerate(candidates)
            if normalized and normalized in normalize_text(candidate["code_body"])
        }
    return set()


def coerce_indices(value: Any) -> set[int]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return {int(stripped)}
        return set()
    if isinstance(value, list):
        indices: set[int] = set()
        for item in value:
            indices.update(coerce_indices(item))
        return indices
    return set()


def is_binary_label_list(value: Any, expected_length: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_length:
        return False
    return all(item in (0, 1, False, True) for item in value)


def build_query(record: dict[str, Any], *, max_chars: int) -> str:
    parts: list[str] = []
    for field in (
        "repo",
        "repo_name",
        "path",
        "file_path",
        "import_statement",
        "imports",
        "cropped_code",
        "in_file_context",
        "infile_context",
        "prompt",
        "query",
        "code",
        "current_file_content",
        "file_content",
    ):
        value = record.get(field)
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if text.strip():
            parts.append(f"{field}:\n{text.strip()}")

    query = "\n\n".join(parts).strip()
    if len(query) <= max_chars:
        return query
    return query[-max_chars:]


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def first_present(item: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        value = item.get(field)
        if value is not None:
            return value
    return None


def make_chunk_id(repo_name: str, query_id: str, candidate_index: int) -> str:
    safe_query_id = str(query_id).replace(":", "_").replace("/", "_").replace("\\", "_")
    return f"{repo_name}:{safe_query_id}:candidate:{candidate_index}"


def infer_language(path: str) -> str:
    if path.endswith(".java"):
        return "java"
    if path.endswith((".ts", ".tsx")):
        return "typescript"
    return "python"


def line_count(text: str) -> int:
    return max(1, text.count("\n") + 1)


def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

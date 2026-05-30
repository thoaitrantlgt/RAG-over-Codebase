import re
import os
from dataclasses import replace
from typing import Any

from packages.indexing.schema import SearchResult


def rerank_results(
    query: str,
    results: list[SearchResult],
    *,
    method: str,
    bge_model: str = "BAAI/bge-reranker-v2-m3",
    bge_cache_dir: str | None = None,
    bge_batch_size: int = 8,
    bge_max_passage_chars: int = 4000,
    bge_use_fp16: bool = False,
) -> list[SearchResult]:
    if method == "none":
        return results
    if method == "lexical":
        return lexical_rerank(query, results)
    if method == "bge-m3":
        return bge_m3_rerank(
            query,
            results,
            model_name=bge_model,
            cache_dir=bge_cache_dir,
            batch_size=bge_batch_size,
            max_passage_chars=bge_max_passage_chars,
            use_fp16=bge_use_fp16,
        )
    raise ValueError(f"Unsupported rerank method: {method}")


def lexical_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    query_tokens = tokenize(query)
    import_names = extract_import_names(query)
    scored: list[SearchResult] = []

    for rank, result in enumerate(results, start=1):
        candidate_text = "\n".join(
            [
                result.symbol_name,
                result.path,
                result.summary,
                result.code_body,
            ]
        )
        candidate_tokens = tokenize(candidate_text)
        score = lexical_score(
            query_tokens=query_tokens,
            candidate_tokens=candidate_tokens,
            import_names=import_names,
            result=result,
            original_rank=rank,
        )
        scored.append(replace(result, score=score, source=f"{result.source}+lexical_rerank"))

    scored.sort(key=lambda item: (-item.score, item.path, item.start_line, item.chunk_id))
    return scored


def bge_m3_rerank(
    query: str,
    results: list[SearchResult],
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    cache_dir: str | None = None,
    batch_size: int = 8,
    max_passage_chars: int = 4000,
    use_fp16: bool = False,
    reranker: Any | None = None,
) -> list[SearchResult]:
    if not results:
        return []

    reranker = reranker or load_bge_reranker(
        model_name=model_name,
        cache_dir=cache_dir,
        use_fp16=use_fp16,
    )
    pairs = [
        [query, format_bge_passage(result, max_chars=max_passage_chars)]
        for result in results
    ]
    scores = reranker.compute_score(
        pairs,
        batch_size=batch_size,
        normalize=True,
    )
    if isinstance(scores, (int, float)):
        scores = [float(scores)]

    scored = [
        replace(result, score=float(score), source=f"{result.source}+bge_m3_rerank")
        for result, score in zip(results, scores)
    ]
    scored.sort(key=lambda item: (-item.score, item.path, item.start_line, item.chunk_id))
    return scored


def load_bge_reranker(
    *,
    model_name: str,
    cache_dir: str | None,
    use_fp16: bool,
):
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "BGE rerank requires FlagEmbedding. Install it with: pip install -r requirements.txt"
        ) from exc
    return FlagReranker(model_name, use_fp16=use_fp16)


def format_bge_passage(result: SearchResult, *, max_chars: int) -> str:
    text = "\n".join(
        [
            f"path: {result.path}",
            f"symbol: {result.symbol_name}",
            f"kind: {result.symbol_kind}",
            f"summary: {result.summary}",
            "code:",
            result.code_body,
        ]
    )
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def lexical_score(
    *,
    query_tokens: set[str],
    candidate_tokens: set[str],
    import_names: set[str],
    result: SearchResult,
    original_rank: int,
) -> float:
    if not candidate_tokens:
        overlap = 0.0
    else:
        overlap = len(query_tokens & candidate_tokens) / len(query_tokens or {"_"})

    symbol = result.symbol_name.lower()
    symbol_tokens = tokenize(result.symbol_name)
    symbol_overlap = len(query_tokens & symbol_tokens) / len(symbol_tokens or {"_"})
    import_overlap = len(import_names & candidate_tokens) / len(import_names or {"_"}) if import_names else 0.0
    exact_symbol = 1.0 if symbol and symbol in " ".join(query_tokens) else 0.0
    rank_prior = 1.0 / original_rank

    return (
        2.0 * rank_prior
        + 2.0 * import_overlap
        + 1.5 * symbol_overlap
        + 0.5 * exact_symbol
        + 0.5 * overlap
    )


def extract_import_names(query: str) -> set[str]:
    names: set[str] = set()
    for raw_line in query.splitlines():
        line = raw_line.strip()
        if line.startswith("from ") and " import " in line:
            imported = line.split(" import ", 1)[1]
            names.update(split_import_names(imported))
        elif line.startswith("import "):
            imported = line.removeprefix("import ")
            names.update(split_import_names(imported))
    return {name for name in names if name}


def split_import_names(imported: str) -> set[str]:
    names: set[str] = set()
    for part in imported.split(","):
        name = part.strip().split(" as ", 1)[0].strip()
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            names.add(name.lower())
    return names


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text)
        if token.lower() not in STOP_TOKENS
    }


STOP_TOKENS = {
    "and",
    "args",
    "class",
    "code",
    "def",
    "false",
    "file",
    "from",
    "import",
    "none",
    "path",
    "query",
    "repo",
    "return",
    "self",
    "true",
    "with",
}

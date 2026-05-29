import json
from urllib import request

from packages.indexing.schema import SearchResult

from .retrieval import citation


class LMStudioClient:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def synthesize(self, query: str, context: list[SearchResult]) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 800,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_prompt(query, context),
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def system_prompt() -> str:
    return (
        "You are a codebase RAG assistant. Answer only from the provided context. "
        "Prefer the context whose symbol exactly appears in the question. "
        "Every concrete claim about code must include a citation in the exact format "
        "repo/path:start_line-end_line. Do not use bracket citations like [1]. "
        "Do not invent citations or line ranges. If the context is insufficient, say so plainly."
    )


def user_prompt(query: str, context: list[SearchResult]) -> str:
    blocks = []
    prompt_context = select_prompt_context(query, context, limit=4)
    allowed_citations = [citation(result) for result in prompt_context]
    for index, result in enumerate(prompt_context, start=1):
        blocks.append(
            "\n".join(
                [
                    f"citation: {citation(result)}",
                    f"symbol: {result.symbol_name}",
                    f"kind: {result.symbol_kind}",
                    f"source: {result.source}",
                    f"summary: {result.summary}",
                    "code:",
                    "```",
                    truncate(result.code_body, 1800),
                    "```",
                ]
            )
        )

    return "\n\n".join(
        [
            f"Question: {query}",
            "Allowed citations:",
            "\n".join(f"- {item}" for item in allowed_citations),
            "Context:",
            "\n\n".join(blocks),
            "Answer in Vietnamese. Be concise. Include citations next to the statements they support.",
        ]
    )


def select_prompt_context(
    query: str,
    context: list[SearchResult],
    *,
    limit: int,
) -> list[SearchResult]:
    if len(context) <= limit:
        return context

    query_lower = query.lower()

    def is_exact_symbol_match(result: SearchResult) -> bool:
        symbol = result.symbol_name.lower()
        short_symbol = symbol.rsplit(".", 1)[-1]
        return symbol in query_lower or short_symbol in query_lower

    def is_test_path(result: SearchResult) -> bool:
        normalized = result.path.replace("\\", "/")
        return "/tests/" in normalized or normalized.startswith("tests/")

    exact_matches = [result for result in context if is_exact_symbol_match(result)]
    selected: list[SearchResult] = []
    selected_ids: set[str] = set()

    for result in exact_matches:
        if selected and len(result.code_body) > 20_000:
            continue
        selected.append(result)
        selected_ids.add(result.chunk_id)
        if len(selected) >= limit:
            return selected

    candidates = [
        result
        for result in context
        if result.chunk_id not in selected_ids
        and not is_test_path(result)
        and len(result.code_body) <= 20_000
    ]
    candidates.extend(
        result
        for result in context
        if result.chunk_id not in selected_ids
        and result not in candidates
        and len(result.code_body) <= 20_000
    )
    candidates.extend(
        result
        for result in context
        if result.chunk_id not in selected_ids and result not in candidates
    )

    for result in candidates:
        if result.chunk_id in selected_ids:
            continue
        selected.append(result)
        selected_ids.add(result.chunk_id)
        if len(selected) >= limit:
            return selected

    return selected


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... [truncated]"

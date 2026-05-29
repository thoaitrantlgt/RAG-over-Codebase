import json
from pathlib import Path
from typing import Protocol

from .text import compact_text, tokenize


class SummaryProvider(Protocol):
    def summarize(self, chunk: dict) -> str:
        ...


class HeuristicSummarizer:
    def summarize(self, chunk: dict) -> str:
        symbol = chunk["symbol_name"]
        kind = chunk.get("symbol_kind", "symbol")
        language = chunk.get("language", "code")
        tokens = tokenize(f"{symbol} {chunk.get('code_body', '')}")
        keywords = unique_ordered(tokens)[:8]
        keyword_text = ", ".join(keywords) if keywords else "implementation details"
        return compact_text(
            f"{kind} {symbol} in {language} handles {keyword_text}.",
            32,
        )


class CachedSummarizer:
    def __init__(self, cache_path: str | Path, provider: SummaryProvider | None = None) -> None:
        self.cache_path = Path(cache_path).resolve()
        self.provider = provider or HeuristicSummarizer()
        self._cache = self._load()

    def summarize(self, chunk: dict) -> str:
        key = chunk["content_hash"]
        if key not in self._cache:
            self._cache[key] = self.provider.summarize(chunk)
        return self._cache[key]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))


def unique_ordered(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result

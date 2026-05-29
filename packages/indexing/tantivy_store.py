import json
import shutil
from pathlib import Path

import tantivy

from .schema import IndexedChunk, SearchResult

SEARCH_FIELDS = ["symbol_name", "summary", "code_body"]
FIELD_BOOSTS = {
    "symbol_name": 4.0,
    "summary": 2.0,
    "code_body": 1.0,
}


class TantivySparseStore:
    def __init__(self, path: str | Path, reset: bool = False) -> None:
        self.path = Path(path).resolve()
        if reset and self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.schema = build_schema()
        self.index = tantivy.Index(self.schema, path=str(self.path), reuse=True)

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        writer = self.index.writer()
        for chunk in chunks:
            writer.delete_documents("chunk_id", chunk.chunk_id)
            writer.add_document(
                tantivy.Document(
                    chunk_id=chunk.chunk_id,
                    symbol_name=chunk.symbol_name,
                    summary=chunk.summary,
                    code_body=chunk.code_body,
                    metadata=json.dumps(chunk.to_dict(), ensure_ascii=False),
                )
            )
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        parsed_query, _ = self.index.parse_query_lenient(
            query,
            SEARCH_FIELDS,
            field_boosts=FIELD_BOOSTS,
        )
        searcher = self.index.searcher()
        search_result = searcher.search(parsed_query, limit=top_k)
        results: list[SearchResult] = []

        for score, doc_address in search_result.hits:
            document = searcher.doc(doc_address).to_dict()
            metadata = document["metadata"][0]
            chunk = json.loads(metadata)
            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    repo=chunk["repo"],
                    path=chunk["path"],
                    start_line=chunk["start_line"],
                    end_line=chunk["end_line"],
                    symbol_name=chunk["symbol_name"],
                    symbol_kind=chunk["symbol_kind"],
                    language=chunk["language"],
                    code_body=chunk["code_body"],
                    summary=chunk["summary"],
                    score=float(score),
                    source="tantivy",
                )
            )
        return results


def build_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("symbol_name", stored=True)
    builder.add_text_field("summary", stored=True)
    builder.add_text_field("code_body", stored=True)
    builder.add_text_field("metadata", stored=True, tokenizer_name="raw")
    return builder.build()

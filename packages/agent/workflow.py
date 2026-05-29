from dataclasses import dataclass
from typing import Protocol

from packages.indexing.schema import SearchResult

from .config import AgentConfig
from .langchain_workflow import LangChainSynthesizer
from .retrieval import citation, expand_retrieved_context, reciprocal_rank_fusion, retrieve


class Synthesizer(Protocol):
    def synthesize(self, query: str, context: list[SearchResult]) -> str:
        ...


@dataclass(frozen=True)
class AgentAnswer:
    query: str
    answer: str
    citations: list[str]
    dense_results: list[SearchResult]
    sparse_results: list[SearchResult]
    fused_results: list[SearchResult]
    context: list[SearchResult]

    def to_dict(self, *, include_debug: bool = False) -> dict:
        payload = {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "context": [result.to_dict() for result in self.context],
        }
        if include_debug:
            payload["debug"] = {
                "dense": [result.to_dict() for result in self.dense_results],
                "sparse": [result.to_dict() for result in self.sparse_results],
                "fused": [result.to_dict() for result in self.fused_results],
            }
        return payload


def run_agent(
    query: str,
    config: AgentConfig,
    *,
    synthesizer: Synthesizer | None = None,
    synthesize: bool = True,
) -> AgentAnswer:
    dense, sparse = retrieve(query, config)
    fused = reciprocal_rank_fusion(
        [dense, sparse],
        limit=config.fused_top_k,
        k=config.rrf_k,
    )
    context = expand_retrieved_context(fused, config)
    citations = [citation(result) for result in context]

    if not synthesize:
        answer = "Synthesis skipped. Retrieved context is available in the response."
    else:
        client = synthesizer or LangChainSynthesizer(
            base_url=config.lmstudio_base_url,
            model=config.lmstudio_model,
            timeout_seconds=config.lmstudio_timeout_seconds,
        )
        answer = client.synthesize(query, context)

    return AgentAnswer(
        query=query,
        answer=answer,
        citations=citations,
        dense_results=dense,
        sparse_results=sparse,
        fused_results=fused,
        context=context,
    )

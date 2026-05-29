from packages.indexing.schema import SearchResult

from .lmstudio import select_prompt_context, truncate
from .retrieval import citation


class LangChainSynthesizer:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: int = 120) -> None:
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LangChain Phase 4 requires langchain-core and langchain-openai. "
                "Install them with: pip install -r requirements.txt"
            ) from exc

        llm = ChatOpenAI(
            base_url=base_url.rstrip("/"),
            api_key="lm-studio",
            model=model,
            temperature=0.2,
            max_tokens=800,
            timeout=timeout_seconds,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt()),
                ("user", user_prompt_template()),
            ]
        )
        self.chain = prompt | llm | StrOutputParser()

    def synthesize(self, query: str, context: list[SearchResult]) -> str:
        prompt_context = select_prompt_context(query, context, limit=4)
        allowed_citations = [citation(result) for result in prompt_context]
        return self.chain.invoke(
            {
                "question": query,
                "allowed_citations": "\n".join(f"- {item}" for item in allowed_citations),
                "context": format_context(prompt_context),
            }
        ).strip()


def system_prompt() -> str:
    return (
        "You are a codebase RAG assistant. Answer only from the provided context. "
        "Prefer the context whose symbol exactly appears in the question. "
        "Every concrete claim about code must include a citation in the exact format "
        "repo/path:start_line-end_line. Do not use bracket citations like [1]. "
        "Do not invent citations or line ranges. If the context is insufficient, say so plainly."
    )


def user_prompt_template() -> str:
    return "\n\n".join(
        [
            "Question: {question}",
            "Allowed citations:",
            "{allowed_citations}",
            "Context:",
            "{context}",
            "Answer in Vietnamese. Be concise. Include citations next to the statements they support.",
        ]
    )


def format_context(context: list[SearchResult]) -> str:
    blocks = []
    for result in context:
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
    return "\n\n".join(blocks)

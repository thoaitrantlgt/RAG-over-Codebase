import re

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        for part in CAMEL_RE.sub(" ", raw.replace("_", " ")).split():
            token = part.lower()
            if token:
                tokens.append(token)
    return tokens


def compact_text(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])

import re

from .config import CONFIG


INDEX_FILE = CONFIG.index_dir / "github" / "ollama_embeddings_index.pkl"
BM25_INDEX_FILE = CONFIG.index_dir / "github" / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    text = text.lower()

    raw_tokens = re.findall(
        r"[a-z0-9_./:-]+|[\u4e00-\u9fff]+",
        text
    )

    tokens = []

    for token in raw_tokens:
        tokens.append(token)

        parts = re.split(r"[./:_\-]+", token)
        tokens.extend(part for part in parts if part)

    return tokens

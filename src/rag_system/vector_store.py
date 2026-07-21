from typing import Any

from .config import CONFIG


def save_chroma_index(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Chroma is enabled but not installed. Run `pip install chromadb`, "
            "or set CHROMA_ENABLED=false."
        ) from exc

    client = chromadb.PersistentClient(path=str(CONFIG.chroma_path))
    collection = client.get_or_create_collection(CONFIG.chroma_collection)

    for start in range(0, len(ids), CONFIG.ollama_embed_batch_size):
        end = start + CONFIG.ollama_embed_batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

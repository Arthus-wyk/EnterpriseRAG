import pickle

from .config import CONFIG
from .embeddings import OllamaEmbeddingClient
from .loader import load_documents
from .retrieval_utils import INDEX_FILE, BM25_INDEX_FILE, tokenize
from .splitter import split_documents
from .vector_store import save_chroma_index
from rank_bm25 import BM25Okapi


def build_bm25_text(chunk) -> str:
    title = chunk.title
    path = chunk.metadata.get("source_relative_path", "")
    source_type = chunk.metadata.get("source_type", "")
    folder_path = chunk.metadata.get("folder_path", "")
    text = chunk.text

    return "\n".join(
        part for part in [
            f"Title: {title}",
            f"Title: {title}",
            f"Source Type: {source_type}",
            f"Folder: {folder_path}",
            f"Folder: {folder_path}",
            f"Path: {path}",
            f"Text: {text}",
        ]
        if part.strip()
    )

def build_index() -> None:
    print("[index] preparing index directories", flush=True)
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[index] loading documents from {CONFIG.raw_data_dir}", flush=True)
    documents = load_documents(CONFIG.raw_data_dir, progress_every=500)
    if not documents:
        raise RuntimeError(f"No .txt or .md documents found in {CONFIG.raw_data_dir}")
    print(f"[index] loaded {len(documents)} documents", flush=True)

    print(
        f"[index] splitting documents, chunk_size={CONFIG.chunk_size}, "
        f"chunk_overlap={CONFIG.chunk_overlap}",
        flush=True,
    )
    chunks = split_documents(
        documents,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    print(f"[index] created {len(chunks)} chunks", flush=True)

    print("[index] formatting chunk texts for embeddings", flush=True)
    chunk_texts = [
        "\n".join(
            part
            for part in [
                f"Title: {chunk.title}",
                f"Source type: {chunk.metadata.get('source_type', '')}",
                f"Folder: {chunk.metadata.get('folder_path', '')}",
                f"Text: {chunk.text}",
            ]
            if part.strip()
        )
        for chunk in chunks
    ]

    print(
        f"[index] embedding {len(chunk_texts)} chunks with "
        f"{CONFIG.model_provider}/{CONFIG.ollama_embed_model}, "
        f"batch_size={CONFIG.ollama_embed_batch_size}",
        flush=True,
    )
    embedder = OllamaEmbeddingClient()
    embeddings = embedder.embed_texts(
        chunk_texts,
        progress_label="embed",
        total=len(chunk_texts),
    )
    print(f"[index] finished embeddings: {len(embeddings)} vectors", flush=True)

    print("[index] building BM25 index", flush=True)
    bm25_texts = [build_bm25_text(chunk) for chunk in chunks]
    tokenized_docs = [tokenize(text) for text in bm25_texts]
    bm25 = BM25Okapi(tokenized_docs)
    print(f"[index] BM25 indexed {len(tokenized_docs)} documents", flush=True)

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    payload = {
        "embedding_provider": CONFIG.model_provider,
        "embedding_model": CONFIG.ollama_embed_model,
        "embedding_base_url": CONFIG.ollama_effective_base_url,
        "ids": ids,
        "documents": chunk_texts,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "chunks": chunks,
    }
    bm25_payload = {
        "ids": ids,
        "documents": bm25_texts,
        "original_documents": chunk_texts,
        "metadatas": metadatas,
        "bm25": bm25,
        "tokenizer": "lower_split",
    }

    print(f"[index] saving vector payload to {INDEX_FILE}", flush=True)
    with INDEX_FILE.open("wb") as file:
        pickle.dump(payload, file)

    print(f"[index] saving BM25 payload to {BM25_INDEX_FILE}", flush=True)
    with BM25_INDEX_FILE.open("wb") as f:
        pickle.dump(bm25_payload, f)

    if CONFIG.chroma_enabled:
        print(f"[index] saving Chroma collection to {CONFIG.chroma_path}", flush=True)
        save_chroma_index(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")
    print(f"Embedding provider: {CONFIG.model_provider}")
    print(f"Embedding model: {CONFIG.ollama_embed_model}")
    print(f"Saved index to {INDEX_FILE}")
    if CONFIG.chroma_enabled:
        print(f"Saved Chroma collection to {CONFIG.chroma_path}")


if __name__ == "__main__":
    build_index()

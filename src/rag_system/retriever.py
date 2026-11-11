import pickle
from dataclasses import dataclass

from config import CONFIG
from embeddings import OllamaEmbeddingClient
from retrieval_utils import INDEX_FILE, BM25_INDEX_FILE, tokenize
from splitter import Chunk


@dataclass(frozen=True)
class RetrievalResults:
    bm25_results: list[tuple[Chunk, float]]
    vector_results: list[tuple[Chunk, float]]
    candidates: list[tuple[Chunk, float]]

    @property
    def rrf_results(self) -> list[tuple[Chunk, float]]:
        return self.candidates


class Retriever:
    def __init__(self) -> None:
        if not INDEX_FILE.exists():
            raise RuntimeError(
                f"Index not found at {INDEX_FILE}. Run: python -m src.rag_system.indexer"
            )

        with INDEX_FILE.open("rb") as file:
            payload = pickle.load(file)
        with BM25_INDEX_FILE.open("rb") as f:
            bm25_payload = pickle.load(f)

        self.matrix = payload.get("embeddings", payload.get("matrix"))
        self.chunks: list[Chunk] = payload["chunks"]
        self.embedder = OllamaEmbeddingClient(
            model=payload["embedding_model"],
            base_url=payload.get("embedding_base_url"),
            provider=payload.get("embedding_provider"),
        )

        self.bm25 = bm25_payload["bm25"]
        self.bm25_ids = bm25_payload["ids"]
        self.bm25_documents = bm25_payload["documents"]
        self.bm25_metadatas = bm25_payload["metadatas"]

    def vector_search(self,query:str, candidate_k:int) -> list[tuple[int, float]]:
        query_vector = self.embedder.embed_query(query)
        scores = [
            sum(chunk_value * query_value for chunk_value, query_value in zip(vector, query_vector))
            for vector in self.matrix
        ]
        top_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:candidate_k]
        return [(index, float(scores[index])) for index in top_indices]

    def bm25_search(self,query:str, candidate_k:int) -> list[tuple[int, float]]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:candidate_k]
        return [(index, float(scores[index])) for index in top_indices]

    def folder_boost_score(self, query: str, chunk: Chunk) -> float:
        query_tokens = {
            token
            for token in tokenize(query)
            if len(token) > 2
        }
        if not query_tokens:
            return 0.0

        folder_text = " ".join(
            str(chunk.metadata.get(key, ""))
            for key in [
                "folder_path",
                "folder_l1",
                "folder_l2",
                "folder_l3",
                "folder_l4",
                "folder_l5",
            ]
        )
        folder_tokens = {
            token
            for token in tokenize(folder_text)
            if len(token) > 2
        }
        if not folder_tokens:
            return 0.0

        overlap = query_tokens & folder_tokens
        return min(len(overlap) / len(query_tokens), 1.0)

    def search_with_stages(
            self,
            query: str,
            bm25_query: str | None = None,
            vector_query: str | None = None,
            bm25_top_k: int | None = None,
            vector_top_k: int | None = None,
            rrf_top_k: int | None = None,

    ) -> RetrievalResults:
        bm25_k = bm25_top_k or CONFIG.bm25_top_k
        vector_k = vector_top_k or CONFIG.vector_top_k

        bm25_results = self.bm25_search(query, bm25_k)
        if bm25_query and bm25_query != query:
            bm25_results += self.bm25_search(bm25_query, bm25_k)

        vector_results = self.vector_search(query, vector_k)
        if vector_query and vector_query != query:
            vector_results += self.vector_search(vector_query, vector_k)

        def unique_chunks(results: list[tuple[int, float]]) -> list[tuple[Chunk, float]]:
            unique: dict[str, tuple[Chunk, float]] = {}
            for index, score in results:
                chunk = self.chunks[index]
                unique.setdefault(chunk.metadata["chunk_id"], (chunk, float(score)))
            return list(unique.values())

        bm25_chunks = unique_chunks(bm25_results)
        vector_chunks = unique_chunks(vector_results)
        candidates = unique_chunks(bm25_results + vector_results)

        return RetrievalResults(
            bm25_results=bm25_chunks,
            vector_results=vector_chunks,
            candidates=candidates,
        )

    def search(
            self,
            query: str,
            rewrite_query: dict[str, str],
            top_k: int | None = None,
            bm25_top_k: int | None = None,
            vector_top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        return self.search_with_stages(
            query=query,
            bm25_query=rewrite_query["keyword_query"],
            vector_query=rewrite_query["semantic_query"],
            bm25_top_k=bm25_top_k,
            vector_top_k=vector_top_k,
            rrf_top_k=top_k,
        ).candidates

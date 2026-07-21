from __future__ import annotations

from dataclasses import dataclass


from src.rag_system.splitter import Chunk


@dataclass
class RerankResult:
    chunk: Chunk
    retrieval_score: float
    rerank_score: float
    final_score: float


class BGEReranker:
    """
    使用 BGE reranker 对第一轮检索结果进行重排序。

    输入:
        query: 用户问题 / 需求文本
        candidates: [(chunk, retrieval_score), ...]

    输出:
        [(chunk, rerank_score), ...]
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-large",
            batch_size: int = 8,
            use_fp16: bool | None = None,
            rerank_weight: float = 1,
            retrieval_weight: float = 0,
    ) -> None:
        from FlagEmbedding import FlagReranker

        if use_fp16 is None:
            use_fp16 = self._has_cuda()

        self.model_name = model_name
        self.batch_size = batch_size
        self.rerank_weight = rerank_weight
        self.retrieval_weight = retrieval_weight

        self.reranker = FlagReranker(
            model_name,
            use_fp16=use_fp16,
        )

    def rerank(
            self,
            query: str,
            candidates: list[tuple[Chunk, float]],
            top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        返回格式:
            [(chunk, rerank_score), ...]
        """

        if not query.strip():
            return candidates[:top_k]

        if not candidates:
            return []

        pairs = []

        for chunk, _retrieval_score in candidates:
            chunk_text = self._format_chunk(chunk)
            pairs.append([query, chunk_text])

        scores = self.reranker.compute_score(
            pairs,
            batch_size=self.batch_size,
        )

        scores = self._normalize_scores(scores)

        rerank_results = self._build_rerank_results(candidates, scores)

        return [
            (item.chunk, item.final_score)
            for item in rerank_results[:top_k]
        ]

    def rerank_with_details(
            self,
            query: str,
            candidates: list[tuple[Chunk, float]],
            top_k: int = 5,
    ) -> list[RerankResult]:
        """
        如果你想同时看 retrieval_score 和 rerank_score，可以用这个方法。
        """

        if not candidates:
            return []

        pairs = []

        for chunk, _retrieval_score in candidates:
            chunk_text = self._format_chunk(chunk)
            pairs.append([query, chunk_text])

        scores = self.reranker.compute_score(
            pairs,
            batch_size=self.batch_size,
        )

        scores = self._normalize_scores(scores)

        rerank_results = self._build_rerank_results(candidates, scores)

        return rerank_results[:top_k]

    def _format_chunk(self, chunk: Chunk) -> str:
        """
        把 chunk 变成 reranker 可以理解的文本。

        这里兼容你的 Chunk 结构:
            chunk.title
            chunk.text
            chunk.metadata
        """

        text = getattr(chunk, "text", "")
        metadata = getattr(chunk, "metadata", {}) or {}
        title = metadata.get("title", "")
        source_type = metadata.get("source_type", "")
        folder_path = metadata.get("folder_path", "")
        relative_path = metadata.get("source_relative_path", "")

        parts = [
            f"Title: {title}" if title else "",
            f"Source_type: {source_type}" if source_type else "",
            f"Folder: {folder_path}" if folder_path else "",
            f"Path: {relative_path}" if relative_path else "",
            f"Text: {text}" if text else "",
        ]

        return "\n".join(part for part in parts if part.strip())

    def _build_rerank_results(
            self,
            candidates: list[tuple[Chunk, float]],
            rerank_scores: list[float],
    ) -> list[RerankResult]:
        retrieval_scores = [float(score) for _chunk, score in candidates]
        normalized_retrieval_scores = self._minmax_scores(retrieval_scores)
        normalized_rerank_scores = self._minmax_scores(rerank_scores)

        rerank_results: list[RerankResult] = []

        for index, ((chunk, retrieval_score), rerank_score) in enumerate(
                zip(candidates, rerank_scores)
        ):
            final_score = (
                self.rerank_weight * normalized_rerank_scores[index]
                + self.retrieval_weight * normalized_retrieval_scores[index]
            )
            rerank_results.append(
                RerankResult(
                    chunk=chunk,
                    retrieval_score=float(retrieval_score),
                    rerank_score=float(rerank_score),
                    final_score=float(final_score),
                )
            )

        rerank_results.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        return rerank_results

    def _minmax_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [1.0 for _score in scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]

    def _normalize_scores(self, scores: Chunk) -> list[float]:
        """
        FlagEmbedding 有时返回 float、list、numpy array。
        这里统一转成 list[float]。
        """

        if isinstance(scores, float) or isinstance(scores, int):
            return [float(scores)]

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        return [float(score) for score in scores]

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

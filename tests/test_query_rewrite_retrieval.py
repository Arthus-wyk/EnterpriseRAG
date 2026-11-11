import unittest

from src.rag_system.evaluator import QuestionRecord, evaluate_question
from src.rag_system.retriever import RetrievalResults
from src.rag_system.retriever import Retriever
from src.rag_system.splitter import Chunk


class QueryRewriteRetrievalTest(unittest.TestCase):
    def test_search_with_stages_falls_back_to_original_query(self) -> None:
        retriever = Retriever.__new__(Retriever)
        retriever.chunks = [
            Chunk(text="result", metadata={"chunk_id": "chunk-0", "chunk_index": 0})
        ]

        def search(query: str, candidate_k: int) -> list[tuple[int, float]]:
            self.assertEqual("original question", query)
            return [(0, 1.0)]

        retriever.bm25_search = search
        retriever.vector_search = search

        results = retriever.search_with_stages(
            query="original question",
            bm25_top_k=1,
            vector_top_k=1,
            rrf_top_k=1,
        )

        self.assertEqual("result", results.rrf_results[0][0].text)

    def test_keeps_original_and_rewritten_queries_then_deduplicates_by_chunk_id(self) -> None:
        retriever = Retriever.__new__(Retriever)
        retriever.chunks = [
            Chunk(text="shared", metadata={"chunk_id": "shared", "chunk_index": 0}),
            Chunk(text="original keyword", metadata={"chunk_id": "bm25-original", "chunk_index": 1}),
            Chunk(text="rewritten keyword", metadata={"chunk_id": "bm25-rewrite", "chunk_index": 2}),
            Chunk(text="original semantic", metadata={"chunk_id": "vector-original", "chunk_index": 3}),
            Chunk(text="rewritten semantic", metadata={"chunk_id": "vector-rewrite", "chunk_index": 4}),
        ]
        bm25_queries = []
        vector_queries = []

        def bm25_search(query: str, candidate_k: int) -> list[tuple[int, float]]:
            bm25_queries.append(query)
            return {
                "original question": [(0, 2.0), (1, 1.0)],
                "compact keyword query": [(0, 2.0), (2, 1.0)],
            }[query]

        def vector_search(query: str, candidate_k: int) -> list[tuple[int, float]]:
            vector_queries.append(query)
            return {
                "original question": [(0, 0.9), (3, 0.8)],
                "standalone semantic query": [(0, 0.9), (4, 0.8)],
            }[query]

        retriever.bm25_search = bm25_search
        retriever.vector_search = vector_search

        results = retriever.search_with_stages(
            query="original question",
            bm25_query="compact keyword query",
            vector_query="standalone semantic query",
            bm25_top_k=2,
            vector_top_k=2,
            rrf_top_k=1,
        )

        self.assertEqual(["original question", "compact keyword query"], bm25_queries)
        self.assertEqual(["original question", "standalone semantic query"], vector_queries)
        self.assertEqual(
            ["shared", "original keyword", "rewritten keyword"],
            [chunk.text for chunk, _ in results.bm25_results],
        )
        self.assertEqual(
            ["shared", "original semantic", "rewritten semantic"],
            [chunk.text for chunk, _ in results.vector_results],
        )
        self.assertEqual(
            ["shared", "original keyword", "rewritten keyword", "original semantic", "rewritten semantic"],
            [chunk.text for chunk, _ in results.rrf_results],
        )

    def test_evaluator_uses_rewritten_queries_for_their_retrievers(self) -> None:
        chunk = Chunk(
            text="answer evidence",
            metadata={
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "chunk_index": 0,
                "title": "title",
            },
        )

        class FakeRetriever:
            bm25_query = None
            vector_query = None

            def search_with_stages(
                    self,
                    query: str,
                    bm25_query: str | None = None,
                    vector_query: str | None = None,
                    **_kwargs,
            ) -> RetrievalResults:
                self.bm25_query = bm25_query
                self.vector_query = vector_query
                return RetrievalResults([(chunk, 1.0)], [(chunk, 1.0)], [(chunk, 1.0)])

        class FakeGenerator:
            rewrite_calls = 0

            def rewrite_queries(self, question: str) -> dict[str, str]:
                self.rewrite_calls += 1
                return {
                    "keyword_query": "compact keyword query",
                    "semantic_query": "standalone semantic query",
                }

            def generate(self, question: str, contexts) -> str:
                return "answer"

        class FakeReranker:
            query = None

            def rerank(self, query: str, candidates, top_k: int):
                self.query = query
                return candidates[:top_k]

        retriever = FakeRetriever()
        generator = FakeGenerator()
        reranker = FakeReranker()
        record = QuestionRecord(
            question_id="q1",
            question_type="semantic",
            source_types=["github"],
            question="original question",
            gold_answer="answer",
            expected_doc_ids=["doc-1"],
            answer_facts=[],
            raw={},
        )

        evaluate_question(
            record=record,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            bm25_top_k=1,
            vector_top_k=1,
            rrf_top_k=1,
            context_top_k=1,
            include_context_text=False,
        )

        self.assertEqual(1, generator.rewrite_calls)
        self.assertEqual("compact keyword query", retriever.bm25_query)
        self.assertEqual("standalone semantic query", retriever.vector_query)
        self.assertEqual("original question", reranker.query)


if __name__ == "__main__":
    unittest.main()

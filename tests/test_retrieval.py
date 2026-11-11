import json

from src.rag_system.generator import Generator
from src.rag_system.retriever import Retriever


WANTED = {"qst_0207", "qst_0243"}


def best_rank(results, target_indices):
    return next(
        (
            (rank, index, score)
            for rank, (index, score) in enumerate(results, 1)
            if index in target_indices
        ),
        None,
    )


def print_comparison(name, original, rewritten) -> None:
    print(f"{name} original:", original)
    print(f"{name} rewritten:", rewritten)
    if original and rewritten:
        print(f"{name} rank improvement:", original[0] - rewritten[0])


def main() -> None:
    with open("all_documents/questions.jsonl", encoding="utf-8") as file:
        questions = {
            row["question_id"]: row
            for line in file
            if (row := json.loads(line))["question_id"] in WANTED
        }

    generator = Generator()
    retriever = Retriever()
    total = len(retriever.chunks)

    for question_id, row in questions.items():
        question = row["question"]
        rewritten = generator.rewrite_queries(question)
        expected = set(row["expected_doc_ids"])
        target_indices = {
            index
            for index, chunk in enumerate(retriever.chunks)
            if chunk.metadata.get("doc_id") in expected
        }

        original_bm25 = best_rank(
            retriever.bm25_search(question, total),
            target_indices,
        )
        rewritten_bm25 = best_rank(
            retriever.bm25_search(rewritten["keyword_query"], total),
            target_indices,
        )
        original_vector = best_rank(
            retriever.vector_search(question, total),
            target_indices,
        )
        rewritten_vector = best_rank(
            retriever.vector_search(rewritten["semantic_query"], total),
            target_indices,
        )

        print("\n", question_id)
        print("question:", question)
        print("keyword query:", rewritten["keyword_query"])
        print("semantic query:", rewritten["semantic_query"])
        print("expected:", expected)
        print("target chunk count:", len(target_indices))
        print_comparison("BM25", original_bm25, rewritten_bm25)
        print_comparison("Vector", original_vector, rewritten_vector)


if __name__ == "__main__":
    main()

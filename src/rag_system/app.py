import argparse


from src.rag_system.reranker import BGEReranker
from .config import CONFIG
from .generator import Generator
from .retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG index.")
    parser.add_argument("question", help="Question to ask")
    args = parser.parse_args()

    retriever = Retriever()
    reranker = BGEReranker() if CONFIG.rerank_enabled else None
    generator = Generator()
    candidates = retriever.search(args.question)
    contexts = (
        reranker.rerank(args.question, candidates, top_k=CONFIG.context_top_k)
        if CONFIG.rerank_enabled and reranker is not None
        else candidates[:CONFIG.context_top_k]
    )
    answer = generator.generate(args.question, contexts)


if __name__ == "__main__":
    main()

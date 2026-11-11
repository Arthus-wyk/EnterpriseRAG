import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.rag_system.reranker import BGEReranker
from .config import CONFIG, PROJECT_ROOT
from .generator import Generator
from .retriever import Retriever
from .splitter import Chunk

DEFAULT_QUESTIONS_FILE = CONFIG.raw_data_dir / "questions.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "eval_results" / "rag_eval.jsonl"

DATASET_DOC_ID_PATTERN = re.compile(r"(dsid_[A-Za-z0-9]+)")


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    question_type: str
    source_types: list[str]
    question: str
    gold_answer: str
    expected_doc_ids: list[str]
    answer_facts: list[str]
    raw: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    question_id: str
    question_type: str
    source_types: list[str]
    question: str
    answer: str
    gold_answer: str
    answer_facts: list[str]
    expected_doc_ids: list[str]
    bm25_retrieved_doc_ids: list[str]
    bm25_retrieval_hit: bool
    bm25_retrieval_rank: int | None
    vector_retrieved_doc_ids: list[str]
    vector_retrieval_hit: bool
    vector_retrieval_rank: int | None
    rrf_retrieved_doc_ids: list[str]
    rrf_retrieval_hit: bool
    rrf_retrieval_rank: int | None
    retrieved_doc_ids: list[str]
    retrieval_hit: bool
    retrieval_rank: int | None
    exact_match: bool
    gold_answer_in_answer: bool
    contexts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "source_types": self.source_types,
            "question": self.question,
            "answer": self.answer,
            "gold_answer": self.gold_answer,
            "answer_facts": self.answer_facts,
            "expected_doc_ids": self.expected_doc_ids,
            "bm25_retrieved_doc_ids": self.bm25_retrieved_doc_ids,
            "bm25_retrieval_hit": self.bm25_retrieval_hit,
            "bm25_retrieval_rank": self.bm25_retrieval_rank,
            "vector_retrieved_doc_ids": self.vector_retrieved_doc_ids,
            "vector_retrieval_hit": self.vector_retrieval_hit,
            "vector_retrieval_rank": self.vector_retrieval_rank,
            "rrf_retrieved_doc_ids": self.rrf_retrieved_doc_ids,
            "rrf_retrieval_hit": self.rrf_retrieval_hit,
            "rrf_retrieval_rank": self.rrf_retrieval_rank,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "retrieval_hit": self.retrieval_hit,
            "retrieval_rank": self.retrieval_rank,
            "exact_match": self.exact_match,
            "gold_answer_in_answer": self.gold_answer_in_answer,
            "contexts": self.contexts,
        }


def load_questions(path: Path) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc

            records.append(
                QuestionRecord(
                    question_id=str(payload.get("question_id", f"line_{line_number}")),
                    question_type=str(payload.get("question_type", "")),
                    source_types=[str(item) for item in payload.get("source_types", [])],
                    question=str(payload.get("question", "")),
                    gold_answer=str(payload.get("gold_answer", "")),
                    expected_doc_ids=[str(item) for item in payload.get("expected_doc_ids", [])],
                    answer_facts=[str(item) for item in payload.get("answer_facts", [])],
                    raw=payload,
                )
            )

    return records


def split_filter_values(values: list[str] | None) -> set[str] | None:
    if not values:
        return None

    tokens: set[str] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                tokens.add(token)

    return tokens or None


def source_type_matches(
        record_source_types: list[str],
        wanted_source_types: set[str] | None,
        match_mode: str,
) -> bool:
    if not wanted_source_types:
        return True

    record_types = set(record_source_types)
    if match_mode == "all":
        return wanted_source_types.issubset(record_types)
    if match_mode == "exact":
        return record_types == wanted_source_types

    return bool(record_types & wanted_source_types)


def filter_questions(
        records: Iterable[QuestionRecord],
        source_types: set[str] | None,
        source_match: str,
        question_types: set[str] | None,
        limit: int | None,
) -> list[QuestionRecord]:
    filtered: list[QuestionRecord] = []

    for record in records:
        if not source_type_matches(record.source_types, source_types, source_match):
            continue
        if question_types and record.question_type not in question_types:
            continue

        filtered.append(record)
        if limit is not None and len(filtered) >= limit:
            break

    return filtered


def extract_dataset_doc_id(chunk: Chunk) -> str:
    candidates = [
        str(chunk.metadata.get("dataset_doc_id", "")),
        str(chunk.metadata.get("doc_id", "")),
        str(chunk.metadata.get("source_relative_path", "")),
        str(chunk.metadata.get("source_path", "")),
    ]

    for candidate in candidates:
        match = DATASET_DOC_ID_PATTERN.search(candidate)
        if match:
            return match.group(1)

    return str(chunk.metadata.get("doc_id", ""))


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def expected_doc_rank(context_payloads: list[dict[str, Any]], expected_doc_ids: set[str]) -> int | None:
    return next(
        (
            context["rank"]
            for context in context_payloads
            if str(context["doc_id"]) in expected_doc_ids
        ),
        None,
    )


def retrieval_doc_ids(context_payloads: list[dict[str, Any]]) -> list[str]:
    return unique_in_order(
        str(context["doc_id"]) for context in context_payloads
    )


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(without_punctuation.split())


def build_context_payload(
        contexts: list[tuple[Chunk, float]],
        include_context_text: bool,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    for rank, (chunk, score) in enumerate(contexts, start=1):
        context_payload: dict[str, Any] = {
            "rank": rank,
            "score": score,
            "doc_id": extract_dataset_doc_id(chunk),
            "source_type": chunk.metadata.get("source_type"),
            "folder_path": chunk.metadata.get("folder_path"),
            "source_path": chunk.metadata.get("source_path"),
            "source_relative_path": chunk.metadata.get("source_relative_path"),
            "title": chunk.title,
            "chunk_index": chunk.index,
        }
        if include_context_text:
            context_payload["text"] = chunk.text
        else:
            context_payload["text_preview"] = chunk.text[:500]

        payloads.append(context_payload)

    return payloads


def evaluate_question(
        record: QuestionRecord,
        retriever: Retriever,
        reranker: BGEReranker | None,
        generator: Generator,
        bm25_top_k: int,
        vector_top_k: int,
        rrf_top_k: int,
        context_top_k: int,
        include_context_text: bool,
) -> EvaluationResult:
    print(f"[eval] retrieving {record.question_id}", flush=True)
    rewritten_queries = generator.rewrite_queries(record.question)
    stage_results = retriever.search_with_stages(
        query=record.question,
        bm25_query=rewritten_queries["keyword_query"],
        vector_query=rewritten_queries["semantic_query"],
        bm25_top_k=bm25_top_k,
        vector_top_k=vector_top_k,
        rrf_top_k=rrf_top_k,
    )
    candidates = stage_results.candidates
    print(
        f"[eval] {record.question_id}: candidates "
        f"bm25={len(stage_results.bm25_results)}, "
        f"vector={len(stage_results.vector_results)}, merged={len(candidates)}",
        flush=True,
    )
    contexts = (
        reranker.rerank(record.question, candidates, top_k=context_top_k)
        if CONFIG.rerank_enabled and reranker is not None
        else candidates[:context_top_k]
    )
    print(f"[eval] {record.question_id}: generating answer from {len(contexts)} contexts", flush=True)
    answer = generator.generate(record.question, contexts)
    bm25_payloads = build_context_payload(stage_results.bm25_results, include_context_text=False)
    vector_payloads = build_context_payload(stage_results.vector_results, include_context_text=False)
    rrf_payloads = build_context_payload(candidates, include_context_text=False)
    context_payloads = build_context_payload(contexts, include_context_text)
    expected_doc_ids = set(record.expected_doc_ids)
    bm25_retrieved_doc_ids = retrieval_doc_ids(bm25_payloads)
    bm25_retrieval_rank = expected_doc_rank(bm25_payloads, expected_doc_ids)
    vector_retrieved_doc_ids = retrieval_doc_ids(vector_payloads)
    vector_retrieval_rank = expected_doc_rank(vector_payloads, expected_doc_ids)
    rrf_retrieved_doc_ids = retrieval_doc_ids(rrf_payloads)
    rrf_retrieval_rank = expected_doc_rank(rrf_payloads, expected_doc_ids)
    retrieved_doc_ids = retrieval_doc_ids(context_payloads)
    retrieval_rank = expected_doc_rank(context_payloads, expected_doc_ids)
    normalized_answer = normalize_answer(answer)
    normalized_gold = normalize_answer(record.gold_answer)

    result = EvaluationResult(
        question_id=record.question_id,
        question_type=record.question_type,
        source_types=record.source_types,
        question=record.question,
        answer=answer,
        gold_answer=record.gold_answer,
        answer_facts=record.answer_facts,
        expected_doc_ids=record.expected_doc_ids,
        bm25_retrieved_doc_ids=bm25_retrieved_doc_ids,
        bm25_retrieval_hit=bm25_retrieval_rank is not None,
        bm25_retrieval_rank=bm25_retrieval_rank,
        vector_retrieved_doc_ids=vector_retrieved_doc_ids,
        vector_retrieval_hit=vector_retrieval_rank is not None,
        vector_retrieval_rank=vector_retrieval_rank,
        rrf_retrieved_doc_ids=rrf_retrieved_doc_ids,
        rrf_retrieval_hit=rrf_retrieval_rank is not None,
        rrf_retrieval_rank=rrf_retrieval_rank,
        retrieved_doc_ids=retrieved_doc_ids,
        retrieval_hit=retrieval_rank is not None,
        retrieval_rank=retrieval_rank,
        exact_match=normalized_answer == normalized_gold,
        gold_answer_in_answer=bool(normalized_gold and normalized_gold in normalized_answer),
        contexts=context_payloads,
    )
    print(
        f"[eval] {record.question_id}: "
        f"bm25_hit={result.bm25_retrieval_hit} "
        f"vector_hit={result.vector_retrieval_hit} "
        f"rrf_hit={result.rrf_retrieval_hit} "
        f"final_hit={result.retrieval_hit} "
        f"final_rank={result.retrieval_rank}",
        flush=True,
    )
    return result


def summarize_results(results: list[EvaluationResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "bm25_retrieval_hit_rate": 0.0,
            "vector_retrieval_hit_rate": 0.0,
            "rrf_retrieval_hit_rate": 0.0,
            "retrieval_hit_rate": 0.0,
            "exact_match_rate": 0.0,
            "gold_answer_in_answer_rate": 0.0,
        }

    bm25_retrieval_hits = sum(1 for result in results if result.bm25_retrieval_hit)
    vector_retrieval_hits = sum(1 for result in results if result.vector_retrieval_hit)
    rrf_retrieval_hits = sum(1 for result in results if result.rrf_retrieval_hit)
    retrieval_hits = sum(1 for result in results if result.retrieval_hit)
    exact_matches = sum(1 for result in results if result.exact_match)
    gold_contains = sum(1 for result in results if result.gold_answer_in_answer)

    return {
        "total": total,
        "bm25_retrieval_hits": bm25_retrieval_hits,
        "bm25_retrieval_hit_rate": bm25_retrieval_hits / total,
        "vector_retrieval_hits": vector_retrieval_hits,
        "vector_retrieval_hit_rate": vector_retrieval_hits / total,
        "rrf_retrieval_hits": rrf_retrieval_hits,
        "rrf_retrieval_hit_rate": rrf_retrieval_hits / total,
        "retrieval_hits": retrieval_hits,
        "retrieval_hit_rate": retrieval_hits / total,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / total,
        "gold_answer_in_answer": gold_contains,
        "gold_answer_in_answer_rate": gold_contains / total,
    }


def write_jsonl(path: Path, results: list[EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def write_markdown(path: Path, results: list[EvaluationResult], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Evaluation",
        "",
        "## Summary",
        "",
        f"- Total: {summary['total']}",
        f"- BM25 retrieval hit rate: {summary['bm25_retrieval_hit_rate']:.2%}",
        f"- Vector retrieval hit rate: {summary['vector_retrieval_hit_rate']:.2%}",
        f"- RRF retrieval hit rate: {summary['rrf_retrieval_hit_rate']:.2%}",
        f"- Retrieval hit rate: {summary['retrieval_hit_rate']:.2%}",
        f"- Exact match rate: {summary['exact_match_rate']:.2%}",
        f"- Gold answer in answer rate: {summary['gold_answer_in_answer_rate']:.2%}",
        "",
        "## Results",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"### {result.question_id}",
                "",
                f"- Source types: {', '.join(result.source_types)}",
                f"- BM25 retrieval hit: {result.bm25_retrieval_hit}",
                f"- BM25 retrieval rank: {result.bm25_retrieval_rank}",
                f"- Vector retrieval hit: {result.vector_retrieval_hit}",
                f"- Vector retrieval rank: {result.vector_retrieval_rank}",
                f"- RRF retrieval hit: {result.rrf_retrieval_hit}",
                f"- RRF retrieval rank: {result.rrf_retrieval_rank}",
                f"- Retrieval hit: {result.retrieval_hit}",
                f"- Retrieval rank: {result.retrieval_rank}",
                f"- Expected doc ids: {', '.join(result.expected_doc_ids)}",
                f"- BM25 retrieved doc ids: {', '.join(result.bm25_retrieved_doc_ids)}",
                f"- Vector retrieved doc ids: {', '.join(result.vector_retrieved_doc_ids)}",
                f"- RRF retrieved doc ids: {', '.join(result.rrf_retrieved_doc_ids)}",
                f"- Retrieved doc ids: {', '.join(result.retrieved_doc_ids)}",
                "",
                "**Question**",
                "",
                result.question,
                "",
                "**Answer**",
                "",
                result.answer,
                "",
                "**Golden Answer**",
                "",
                result.gold_answer,
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers against all_documents/questions.jsonl."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help=f"JSONL question file. Default: {DEFAULT_QUESTIONS_FILE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output path. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "markdown"],
        default="jsonl",
        help="Output format.",
    )
    parser.add_argument(
        "--source-types",
        nargs="*",
        help="Filter by source types. Accepts spaces or commas, for example: github slack, or github,slack.",
    )
    parser.add_argument(
        "--source-match",
        choices=["any", "all", "exact"],
        default="exact",
        help="How --source-types should match multi-source questions. Defaults to exact.",
    )
    parser.add_argument(
        "--question-types",
        nargs="*",
        help="Optional filter by question_type. Accepts spaces or commas.",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of questions to run.")
    parser.add_argument(
        "--top-k",
        type=int,
        dest="rrf_top_k",
        help=(
            "Backward-compatible alias for --rrf-top-k. "
            f"Default: {CONFIG.rrf_top_k}."
        ),
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=CONFIG.bm25_top_k,
        help=f"Number of BM25 candidates to fuse with RRF. Default: {CONFIG.bm25_top_k}.",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=CONFIG.vector_top_k,
        help=f"Number of vector candidates to fuse with RRF. Default: {CONFIG.vector_top_k}.",
    )
    parser.add_argument(
        "--rrf-top-k",
        type=int,
        default=CONFIG.rrf_top_k,
        help=f"Number of fused RRF candidates to evaluate and pass to reranking. Default: {CONFIG.rrf_top_k}.",
    )
    parser.add_argument(
        "--context-top-k",
        type=int,
        default=CONFIG.context_top_k,
        help=(
            "Number of final contexts passed to the generator and used for "
            f"retrieval_hit/retrieval_rank. Default: {CONFIG.context_top_k}."
        ),
    )
    parser.add_argument(
        "--include-context-text",
        action="store_true",
        help="Include full retrieved chunk text in the output instead of 500-character previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[eval] loading questions from {args.questions}", flush=True)
    records = load_questions(args.questions)
    print(f"[eval] loaded {len(records)} questions", flush=True)
    source_types = split_filter_values(args.source_types)
    question_types = split_filter_values(args.question_types)
    filtered_records = filter_questions(
        records=records,
        source_types=source_types,
        source_match=args.source_match,
        question_types=question_types,
        limit=args.limit,
    )

    if not filtered_records:
        raise RuntimeError("No questions matched the requested filters.")
    print(f"[eval] evaluating {len(filtered_records)} questions", flush=True)

    print("[eval] loading retriever/index", flush=True)
    retriever = Retriever()
    print("[eval] loading reranker" if CONFIG.rerank_enabled else "[eval] reranker disabled", flush=True)
    reranker = BGEReranker() if CONFIG.rerank_enabled else None
    print("[eval] loading generator", flush=True)
    generator = Generator()
    results: list[EvaluationResult] = []
    for index, record in enumerate(filtered_records, start=1):
        print(
            f"[eval] progress {index}/{len(filtered_records)} "
            f"({index / len(filtered_records) * 100:.1f}%): {record.question_id}",
            flush=True,
        )
        results.append(evaluate_question(
            record=record,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            bm25_top_k=args.bm25_top_k,
            vector_top_k=args.vector_top_k,
            rrf_top_k=args.rrf_top_k,
            context_top_k=args.context_top_k,
            include_context_text=args.include_context_text,
        ))
    summary = summarize_results(results)

    print(f"[eval] writing {args.format} results to {args.output}", flush=True)
    if args.format == "markdown":
        write_markdown(args.output, results, summary)
    else:
        write_jsonl(args.output, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()

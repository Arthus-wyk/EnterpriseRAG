# RAG Evaluation

Run the evaluator against the question set:

```bash
python -m src.rag_system.evaluator --source-types github --output eval_results/github_eval.jsonl
```

The evaluator uses the configured Ollama LLM for answer generation. By default:

```bash
MODEL_PROVIDER=ollama
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

Set `OLLAMA_LLM_MODEL` to any local model you have pulled before running the evaluation.

To use an Ollama-compatible cloud endpoint instead:

```bash
MODEL_PROVIDER=ollama_cloud
OLLAMA_CLOUD_BASE_URL=https://ollama.com
OLLAMA_CLOUD_API_KEY=your_api_key
OLLAMA_CLOUD_REQUESTS_PER_MINUTE=60
OLLAMA_CLOUD_MAX_RETRIES=3
OLLAMA_CLOUD_RETRY_BACKOFF_SECONDS=2
OLLAMA_LLM_MODEL=gpt-oss:120b-cloud
OLLAMA_EMBED_MODEL=bge-large
```

Both local Ollama and Ollama Cloud use the same `/api/generate` and `/api/embed`
request shape. When `MODEL_PROVIDER=ollama_cloud`, requests include
`Authorization: Bearer <OLLAMA_CLOUD_API_KEY>`.
Cloud requests are rate-limited by `OLLAMA_CLOUD_REQUESTS_PER_MINUTE`, and HTTP
429 responses are retried with `Retry-After` when provided.
For `gpt-oss`, Ollama Cloud model tags include `gpt-oss:20b-cloud` and
`gpt-oss:120b-cloud`. The generator also maps local tags `gpt-oss:20b` and
`gpt-oss:120b` to their Cloud equivalents automatically.

For a human-readable report:

```bash
python -m src.rag_system.evaluator --source-types github --format markdown --output eval_results/github_eval.md
```

Useful filters:

```bash
python -m src.rag_system.evaluator --source-types github,slack --source-match any
python -m src.rag_system.evaluator --source-types github --question-types basic --limit 10
```

Retrieval top-k controls:

```bash
python -m src.rag_system.evaluator \
  --source-types github \
  --bm25-top-k 100 \
  --vector-top-k 100 \
  --rrf-top-k 50
```

By default, the evaluator fuses BM25 top 100 and vector top 100 with RRF, then
keeps RRF top 50. It reports gold document hit rates for all three candidate
sets: BM25 top k, vector top k, and RRF top k.

Output fields include:

- `answer`: the LLM-generated answer based on the retrieved context.
- `gold_answer`: the golden answer from `all_documents/questions.jsonl`.
- `expected_doc_ids`: the document ids the dataset expects.
- `bm25_retrieved_doc_ids`, `bm25_retrieval_hit`, and `bm25_retrieval_rank`: whether an expected document appeared in BM25 top k, and the first BM25 rank where it appeared.
- `vector_retrieved_doc_ids`, `vector_retrieval_hit`, and `vector_retrieval_rank`: whether an expected document appeared in vector top k, and the first vector rank where it appeared.
- `rrf_retrieved_doc_ids`, `rrf_retrieval_hit`, and `rrf_retrieval_rank`: whether an expected document appeared in fused RRF top k, and the first RRF rank where it appeared.
- `retrieved_doc_ids`: unique retrieved `dsid_...` ids in rank order.
- `retrieval_hit` and `retrieval_rank`: whether an expected document appeared in the final context chunks passed to the generator, and the first rank where it appeared. By default the evaluator keeps 5 final contexts (`CONTEXT_TOP_K=5`).
- `exact_match` and `gold_answer_in_answer`: simple answer string checks. For semantic grading, use these reports as the input to a stricter judge or fact-level scorer.

The evaluator reads `all_documents/questions.jsonl` by default and supports source filtering with `--source-types`. By default, source filtering is exact: `--source-types github` only includes questions whose `source_types` is exactly `["github"]`, not mixed-source questions like `["github", "jira", "slack"]`. Use `--source-match any` when mixed-source questions should be included.

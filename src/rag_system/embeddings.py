import json
import math
from typing import Iterable
from urllib import error, request

from .config import CONFIG, normalize_model_provider
from .http_client import urlopen_with_provider_limits


class OllamaEmbeddingClient:
    def __init__(
        self,
        model: str = CONFIG.ollama_embed_model,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        batch_size: int = CONFIG.ollama_embed_batch_size,
    ) -> None:
        self.model = model
        self.provider = normalize_model_provider(provider or CONFIG.model_provider)
        self.base_url = (base_url or self._default_base_url()).rstrip("/")
        self.api_key = self._default_api_key() if api_key is None else api_key
        self.batch_size = batch_size

    def embed_texts(
        self,
        texts: Iterable[str],
        progress_label: str | None = None,
        total: int | None = None,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batch: list[str] = []
        embedded_count = 0
        batch_count = 0

        for text in texts:
            batch.append(text)
            if len(batch) >= self.batch_size:
                embeddings.extend(self._embed_batch(batch))
                embedded_count += len(batch)
                batch_count += 1
                self._print_progress(progress_label, embedded_count, total, batch_count)
                batch = []

        if batch:
            embeddings.extend(self._embed_batch(batch))
            embedded_count += len(batch)
            batch_count += 1
            self._print_progress(progress_label, embedded_count, total, batch_count)

        return normalize_embeddings(embeddings)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps(
            {
                "model": self.model,
                "input": texts,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/embed",
            data=body,
            headers=self._headers(),
            method="POST",
        )

        try:
            with urlopen_with_provider_limits(
                    req,
                    provider=self.provider,
                    timeout=CONFIG.ollama_timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to {self.provider}. Check the configured URL "
                f"({self.base_url}) and model ({self.model})."
            ) from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(f"Unexpected Ollama embedding response: {payload}")

        return embeddings

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _print_progress(
        self,
        progress_label: str | None,
        embedded_count: int,
        total: int | None,
        batch_count: int,
    ) -> None:
        if not progress_label:
            return

        if total:
            percent = embedded_count / total * 100
            print(
                f"[{progress_label}] embedded {embedded_count}/{total} texts "
                f"({percent:.1f}%), batches={batch_count}",
                flush=True,
            )
        else:
            print(
                f"[{progress_label}] embedded {embedded_count} texts, batches={batch_count}",
                flush=True,
            )

    def _default_base_url(self) -> str:
        if self.provider == "ollama_cloud":
            return CONFIG.ollama_cloud_base_url

        return CONFIG.ollama_base_url

    def _default_api_key(self) -> str:
        if self.provider == "ollama_cloud":
            return CONFIG.ollama_cloud_api_key

        return CONFIG.ollama_api_key


def normalize_embeddings(matrix: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for vector in matrix:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            normalized.append(vector)
        else:
            normalized.append([value / norm for value in vector])

    return normalized

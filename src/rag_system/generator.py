import json
from urllib import error, request

from .config import CONFIG, normalize_model_provider
from .http_client import urlopen_with_provider_limits
from .splitter import Chunk


SYSTEM_PROMPT = """You are an enterprise RAG answer generator.
Answer the question using only the retrieved context.
If the context does not contain enough evidence, say that the answer is not available in the retrieved context.
Be concise, factual, and include exact names, numbers, and limits when present.
"""


class Generator:
    def __init__(
        self,
        model: str = CONFIG.ollama_llm_model,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.provider = normalize_model_provider(provider or CONFIG.model_provider)
        self.model = self._normalize_model_for_provider(model)
        self.base_url = (base_url or self._default_base_url()).rstrip("/")
        self.api_key = self._default_api_key() if api_key is None else api_key

    def generate(self, question: str, contexts: list[tuple[Chunk, float]]) -> str:
        if not contexts:
            return "No relevant context was retrieved."

        prompt = build_prompt(question, contexts)
        return self._generate_with_ollama(prompt)

    def _generate_with_ollama(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0,
                },
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
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
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            hint = self._model_not_found_hint() if exc.code == 404 else ""
            raise RuntimeError(
                f"Ollama generation request failed with HTTP {exc.code}. "
                f"Provider: {self.provider}. Model: {self.model}. Details: {details}{hint}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to {self.provider} for answer generation. Check "
                f"the configured URL ({self.base_url}) and model ({self.model})."
            ) from exc

        answer = payload.get("response")
        if not isinstance(answer, str):
            raise RuntimeError(f"Unexpected Ollama generation response: {payload}")

        return answer.strip()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _default_base_url(self) -> str:
        if self.provider == "ollama_cloud":
            return CONFIG.ollama_cloud_base_url

        return CONFIG.ollama_base_url

    def _default_api_key(self) -> str:
        if self.provider == "ollama_cloud":
            return CONFIG.ollama_cloud_api_key

        return CONFIG.ollama_api_key

    def _normalize_model_for_provider(self, model: str) -> str:
        if self.provider != "ollama_cloud":
            return model

        cloud_aliases = {
            "gpt-oss:20b": "gpt-oss:20b-cloud",
            "gpt-oss:120b": "gpt-oss:120b-cloud",
        }
        return cloud_aliases.get(model, model)

    def _model_not_found_hint(self) -> str:
        if self.provider != "ollama_cloud":
            return ""

        return (
            " Hint: Ollama Cloud uses cloud model tags such as "
            "`gpt-oss:20b-cloud` or `gpt-oss:120b-cloud`; local tags like "
            "`gpt-oss:120b` may return 404."
        )


def build_prompt(question: str, contexts: list[tuple[Chunk, float]]) -> str:
    context_text = "\n\n".join(
        (
            f"[Context {index}]\n"
            f"Source type: {chunk.metadata.get('source_type', '')}\n"
            f"Folder: {chunk.metadata.get('folder_path', '')}\n"
            f"Path: {chunk.metadata.get('source_relative_path', chunk.source)}\n"
            f"Score: {score:.3f}\n"
            f"Title: {chunk.title}\n"
            f"Text:\n{chunk.text}"
        )
        for index, (chunk, score) in enumerate(contexts, start=1)
    )

    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Answer:"
    )

from dataclasses import dataclass
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv:
    load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "on"}


def normalize_model_provider(value: str) -> str:
    return value.strip().lower().replace("-", "_")


@dataclass(frozen=True)
class RagConfig:
    raw_data_dir: Path = PROJECT_ROOT / "all_documents"
    index_dir: Path = PROJECT_ROOT / "all_documents/github"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "100"))
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "100"))
    rrf_top_k: int = int(os.getenv("RRF_TOP_K", "150"))
    context_top_k: int = int(os.getenv("CONTEXT_TOP_K", "5"))
    folder_boost_enabled: bool = env_bool("FOLDER_BOOST_ENABLED", True)
    folder_boost_weight: float = float(os.getenv("FOLDER_BOOST_WEIGHT", "0.02"))
    model_provider: str = normalize_model_provider(os.getenv("MODEL_PROVIDER", "ollama"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    ollama_cloud_base_url: str = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com/v1")
    ollama_cloud_api_key: str = os.getenv("OLLAMA_CLOUD_API_KEY", "679b5d3ff444466594162284513de227.3O90w3UtgtQTvoMx-CtRDTs3")
    ollama_cloud_requests_per_minute: int = int(os.getenv("OLLAMA_CLOUD_REQUESTS_PER_MINUTE", "60"))
    ollama_cloud_max_retries: int = int(os.getenv("OLLAMA_CLOUD_MAX_RETRIES", "3"))
    ollama_cloud_retry_backoff_seconds: float = float(os.getenv("OLLAMA_CLOUD_RETRY_BACKOFF_SECONDS", "2"))
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-large")
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
    ollama_embed_batch_size: int = int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "16"))
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
    rerank_enabled: bool = env_bool("RERANK_ENABLED", True)
    chroma_enabled: bool = env_bool("CHROMA_ENABLED", False)
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "vector_db")))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "enterprise_rag")

    @property
    def ollama_effective_base_url(self) -> str:
        if self.model_provider == "ollama_cloud":
            return self.ollama_cloud_base_url

        return self.ollama_base_url

    @property
    def ollama_effective_api_key(self) -> str:
        if self.model_provider == "ollama_cloud":
            return self.ollama_cloud_api_key

        return self.ollama_api_key


CONFIG = RagConfig()

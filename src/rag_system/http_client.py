import time
from threading import Lock
from urllib import error, request

from .config import CONFIG


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval_seconds = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.next_allowed_at = 0.0
        self.lock = Lock()

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return

        with self.lock:
            now = time.monotonic()
            wait_seconds = self.next_allowed_at - now
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()

            self.next_allowed_at = now + self.interval_seconds


OLLAMA_CLOUD_RATE_LIMITER = RateLimiter(CONFIG.ollama_cloud_requests_per_minute)


def urlopen_with_provider_limits(
        req: request.Request,
        provider: str,
        timeout: int,
):
    attempts = CONFIG.ollama_cloud_max_retries + 1 if provider == "ollama_cloud" else 1

    for attempt in range(attempts):
        if provider == "ollama_cloud":
            OLLAMA_CLOUD_RATE_LIMITER.wait()

        try:
            return request.urlopen(req, timeout=timeout)
        except error.HTTPError as exc:
            if provider != "ollama_cloud" or exc.code != 429 or attempt >= attempts - 1:
                raise

            time.sleep(_retry_after_seconds(exc, attempt))

    raise RuntimeError("Unexpected retry loop exit.")


def _retry_after_seconds(exc: error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    return CONFIG.ollama_cloud_retry_backoff_seconds * (2 ** attempt)

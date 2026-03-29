import os
from typing import Any

DEFAULT_COLLECTION = "regbot_policy"
# Minimum fraction of recommendation tokens that must appear in at least one cited chunk
# (token recall). Set to 0.0 to disable overlap filtering for the LLM path.
MIN_TOKEN_OVERLAP = float(os.getenv("REGBOT_MIN_TOKEN_OVERLAP", "0.06"))
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "REGBOT_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_LLM_MODEL = os.getenv("REGBOT_LLM_MODEL", "gpt-4o-mini")
CHROMA_SUBDIR = "chroma"
MANIFEST_NAME = "manifest.json"


def _nonneg_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# Retries for transient OpenAI API errors (429, 5xx); see OpenAI client max_retries.
OPENAI_MAX_RETRIES = _nonneg_int_env("REGBOT_OPENAI_MAX_RETRIES", 3)


def chromadb_settings() -> Any:
    """Chroma PersistentClient settings. Telemetry defaults off unless REGBOT_CHROMA_ANONYMIZED_TELEMETRY=1."""
    from chromadb.config import Settings

    raw = os.getenv("REGBOT_CHROMA_ANONYMIZED_TELEMETRY", "0").strip().lower()
    telemetry_on = raw in ("1", "true", "yes", "on")
    return Settings(anonymized_telemetry=telemetry_on)

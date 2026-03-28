import os

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

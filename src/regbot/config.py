import os

DEFAULT_COLLECTION = "regbot_policy"
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "REGBOT_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_LLM_MODEL = os.getenv("REGBOT_LLM_MODEL", "gpt-4o-mini")
CHROMA_SUBDIR = "chroma"
MANIFEST_NAME = "manifest.json"

"""Environment-backed settings and shared Gemini client."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
# Chunk settings are measured in words, not characters.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "false").lower() in {"1", "true", "yes"}

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
INDEX_PATH = PROJECT_ROOT / "storage" / "index.faiss"
CHUNKS_PATH = PROJECT_ROOT / "storage" / "chunks.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemma-4-26b-a4b-it")
EVAL_MAX_INPUT_TOKENS_PER_MIN = int(os.getenv("EVAL_MAX_INPUT_TOKENS_PER_MIN", "15000"))
EVAL_MAX_CALLS_PER_MIN = int(os.getenv("EVAL_MAX_CALLS_PER_MIN", "4"))
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "eval" / "ares_dawn_qa.json"
EVAL_OUTPUT_DIR = OUTPUT_DIR / "eval"
client = genai.Client(api_key=GEMINI_API_KEY)


def require_api_key() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")

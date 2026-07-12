"""
=========================================================
RAG Configuration File
=========================================================

Modify these settings to experiment with different
chunk sizes, embedding models, retrieval parameters,
and language models.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = "data"

INDEX_DIR = "vector_index"

INDEX_PATH = os.path.join(
    INDEX_DIR,
    "faiss.index",
)

META_PATH = os.path.join(
    INDEX_DIR,
    "metadata.pkl",
)

# ==========================================================
# DOCUMENT CHUNKING
# ==========================================================

# Characters per chunk
CHUNK_SIZE = 500

# Shared characters between chunks
CHUNK_OVERLAP = 100

# ==========================================================
# EMBEDDING MODEL
# ==========================================================

"""
Recommended Models

all-MiniLM-L6-v2
    Fast CPU model (384 dimensions)

all-mpnet-base-v2
    Better quality (768 dimensions)

BAAI/bge-small-en-v1.5
    Excellent retrieval quality
"""

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ==========================================================
# VECTOR SEARCH
# ==========================================================

# Number of chunks retrieved
TOP_K = 4

# Future compatibility
SIMILARITY_METRIC = "cosine"

VECTOR_DATABASE = "FAISS"

# ==========================================================
# GENERATION
# ==========================================================

"""
Available Backends

local
openai
anthropic
"""

LLM_BACKEND = os.getenv(
    "LLM_BACKEND",
    "local",
)

# Temperature for supported LLMs
TEMPERATURE = 0.2

# Maximum generated tokens
MAX_NEW_TOKENS = 256

# ==========================================================
# MODEL NAMES
# ==========================================================

ANTHROPIC_MODEL = "claude-sonnet-4-6"

OPENAI_MODEL = "gpt-4o-mini"

LOCAL_MODEL_NAME = "google/flan-t5-base"

# ==========================================================
# UI SETTINGS
# ==========================================================

SHOW_METRICS = True

SHOW_SOURCES = True

SHOW_RETRIEVAL_SCORES = True

# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

SYSTEM_INFO = {
    "Vector Database": VECTOR_DATABASE,
    "Embedding Model": EMBEDDING_MODEL_NAME,
    "Chunk Size": CHUNK_SIZE,
    "Chunk Overlap": CHUNK_OVERLAP,
    "Top K": TOP_K,
    "LLM Backend": LLM_BACKEND,
}
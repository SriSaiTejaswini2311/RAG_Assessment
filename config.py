import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Create directories if they do not exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
CHROMA_DB_DIR.mkdir(exist_ok=True, parents=True)

# API Keys — fallback chain: Streamlit Cloud secrets → .env → empty string
def _get_groq_key() -> str:
    """Read Groq API key from Streamlit secrets (cloud) or .env (local)."""
    # 1. Try Streamlit Cloud secrets first (available when deployed)
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 2. Fallback to environment variable loaded from .env
    return os.getenv("GROQ_API_KEY", "")

GROQ_API_KEY = _get_groq_key()

# Embedding Model Configuration
# Supported models list for streamlit configuration
SUPPORTED_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration
# Groq model
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

# RAG Configuration Defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4

def is_groq_api_key_set() -> bool:
    """Check if the Groq API key is configured and not a placeholder.
    
    Returns:
        bool: True if a valid-looking key exists, False otherwise.
    """
    return bool(GROQ_API_KEY and not GROQ_API_KEY.startswith("your_") and len(GROQ_API_KEY.strip()) > 0)

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

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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

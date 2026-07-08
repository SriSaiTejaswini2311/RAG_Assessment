import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    """Initialize and return a HuggingFaceEmbeddings model.

    Args:
        model_name (str): HuggingFace model identifier (e.g. 'sentence-transformers/all-MiniLM-L6-v2').

    Returns:
        HuggingFaceEmbeddings: The initialized embeddings model.
    """
    logger.info(f"Initializing HuggingFaceEmbeddings model: {model_name}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Successfully initialized HuggingFaceEmbeddings.")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing HuggingFaceEmbeddings model {model_name}: {str(e)}")
        raise e

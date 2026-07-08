import logging
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

def get_retriever(vector_store: Chroma, top_k: int = 4) -> VectorStoreRetriever:
    """Get a retriever interface from the Chroma vector store.

    Args:
        vector_store (Chroma): The vector store to retrieve from.
        top_k (int): Number of top relevant document chunks to return.

    Returns:
        VectorStoreRetriever: The configured LangChain retriever.
    """
    logger.info(f"Configuring retriever with search type 'similarity' and k={top_k}")
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}")
        raise e

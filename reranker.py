import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Cache the CrossEncoder to avoid loading it on every query
_RERANKER_MODEL = None

def get_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Initialize and cache the local CrossEncoder reranking model."""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        try:
            logger.info(f"Loading CrossEncoder reranker model: {model_name}")
            from sentence_transformers import CrossEncoder
            # Load on CPU for universal hardware compatibility
            _RERANKER_MODEL = CrossEncoder(model_name, device="cpu")
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder reranker: {str(e)}")
            raise e
    return _RERANKER_MODEL

def rerank_documents(query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
    """Re-rank retrieved documents using a local CrossEncoder model.

    Args:
        query (str): The user query.
        documents (List[Document]): The retrieved documents from the vector store.
        top_n (int): Number of top documents to return after re-ranking.

    Returns:
        List[Document]: The re-ranked top N documents.
    """
    if not documents:
        return []
    if len(documents) <= 1:
        return documents
        
    try:
        model = get_reranker_model()
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs)
        
        # Sort documents by relevance scores in descending order
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Re-ranked {len(documents)} down to {min(top_n, len(documents))} documents.")
        return [doc for doc, score in scored_docs[:top_n]]
    except Exception as e:
        logger.warning(f"Error during re-ranking: {str(e)}. Falling back to original document order.")
        return documents[:top_n]

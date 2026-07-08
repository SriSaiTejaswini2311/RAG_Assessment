import logging
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import config

logger = logging.getLogger(__name__)

def initialize_vector_store(embeddings: Embeddings) -> Chroma:
    """Initialize a persistent Chroma vector store.

    Args:
        embeddings (Embeddings): The embeddings model to use.

    Returns:
        Chroma: The persistent Chroma vector store instance.
    """
    logger.info(f"Initializing ChromaDB vector store at: {config.CHROMA_DB_DIR}")
    try:
        vector_store = Chroma(
            collection_name="pdf_qa_collection",
            embedding_function=embeddings,
            persist_directory=str(config.CHROMA_DB_DIR)
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}")
        raise e

def add_documents(vector_store: Chroma, documents: List[Document]) -> None:
    """Add a list of document chunks to the Chroma vector store.

    Args:
        vector_store (Chroma): The initialized Chroma instance.
        documents (List[Document]): The document chunks to insert.
    """
    if not documents:
        logger.warning("No documents provided to add to vector store.")
        return
    logger.info(f"Adding {len(documents)} document chunks to ChromaDB.")
    try:
        vector_store.add_documents(documents)
        logger.info("Successfully added documents to ChromaDB.")
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {str(e)}")
        raise e

def clear_vector_store(vector_store: Chroma) -> None:
    """Clear all documents from the Chroma vector store.

    Args:
        vector_store (Chroma): The Chroma instance to clear.
    """
    logger.info("Clearing ChromaDB vector store.")
    try:
        results = vector_store.get()
        ids = results.get("ids", [])
        if ids:
            logger.info(f"Deleting {len(ids)} document IDs from collection.")
            vector_store.delete(ids=ids)
            logger.info("Successfully deleted all documents from collection.")
        else:
            logger.info("Collection is already empty.")
    except Exception as e:
        logger.error(f"Error clearing ChromaDB: {str(e)}")
        raise e

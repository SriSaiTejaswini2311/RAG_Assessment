import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF document using PyPDFLoader.

    Args:
        file_path (str): The absolute or relative path to the PDF file.

    Returns:
        List[Document]: A list of Document objects extracted from the PDF.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any other loading errors.
    """
    logger.info(f"Loading PDF from path: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
        return documents
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"Error loading PDF from {file_path}: {str(e)}")
        raise e

def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """Load multiple PDF documents and combine them into a single list of Documents.

    Args:
        file_paths (List[str]): List of PDF file paths.

    Returns:
        List[Document]: Combined list of Document objects from all PDFs.
    """
    all_documents = []
    for path in file_paths:
        try:
            docs = load_pdf(path)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Skipping {path} due to error: {str(e)}")
    return all_documents

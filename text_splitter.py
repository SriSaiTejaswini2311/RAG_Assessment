import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (List[Document]): The input documents to split.
        chunk_size (int): Target character length of each chunk.
        chunk_overlap (int): Character overlap between consecutive chunks.

    Returns:
        List[Document]: A list of new Document objects representing the chunks.
    """
    logger.info(f"Splitting {len(documents)} document pages (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        # Add explicit metadata for Chunk ID to satisfy ingestion specifications
        for idx, chunk in enumerate(chunks):
            src = chunk.metadata.get('source', 'Unknown').split('\\')[-1].split('/')[-1]
            page = chunk.metadata.get('page', 0) + 1
            chunk.metadata['chunk_id'] = f"{src}_p{page}_c{idx}"
            
        logger.info(f"Split documents into {len(chunks)} chunks with custom chunk IDs.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise e

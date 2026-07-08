import logging
from typing import List, Dict, Any, Generator
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma

import config
from prompts import get_qa_prompt
from retriever import get_retriever

logger = logging.getLogger(__name__)

def get_llm(model_name: str = config.DEFAULT_LLM_MODEL, temperature: float = 0.0) -> ChatGroq:
    """Initialize and return ChatGroq LLM instance.

    Args:
        model_name (str): The name of the Groq model.
        temperature (float): The model temperature.

    Returns:
        ChatGroq: The LLM client.

    Raises:
        ValueError: If the Groq API key is missing.
    """
    logger.info(f"Initializing ChatGroq LLM with model: {model_name}")
    try:
        # Check API key first
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set in the environment variables. "
                "Please add your Groq API key to a .env file."
            )
            
        return ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
            streaming=True
        )
    except Exception as e:
        logger.error(f"Error initializing ChatGroq: {str(e)}")
        raise e

def _enrich_metadata_docs(question: str, vector_store: Chroma, docs: List[Document]) -> List[Document]:
    """Ensure Page 1 (page index 0) of all indexed PDFs is always included in the context
    to resolve document metadata questions (such as case names, delivery dates, or titles).
    """
    try:
        # Query Chroma for the first page (page 0) of all documents in the store
        first_pages_data = vector_store.get(where={"page": 0})
        if first_pages_data and first_pages_data.get("documents"):
            first_page_docs = []
            for text, meta in zip(first_pages_data["documents"], first_pages_data["metadatas"]):
                # Ensure text is a valid string to prevent Pydantic validation errors
                if text is not None:
                    first_page_docs.append(Document(page_content=str(text), metadata=meta))
            
            # Prepend first page of documents if not already present in the retrieved list (checking text content)
            existing_contents = {d.page_content for d in docs}
            for fp_doc in first_page_docs:
                if fp_doc.page_content not in existing_contents:
                    docs.insert(0, fp_doc)
                    existing_contents.add(fp_doc.page_content)
    except Exception as e:
        logger.warning(f"Failed to enrich context with first page: {str(e)}")
        
    return docs

def query_rag_pipeline(
    question: str,
    vector_store: Chroma,
    top_k: int = 4,
    llm_model: str = config.DEFAULT_LLM_MODEL,
    use_reranking: bool = False
) -> Dict[str, Any]:
    """Execute the RAG pipeline synchronously (without streaming).

    Args:
        question (str): User question.
        vector_store (Chroma): The Chroma vector store.
        top_k (int): Number of sources to retrieve.
        llm_model (str): LLM model name.
        use_reranking (bool): Whether to use CrossEncoder re-ranking.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'answer': The LLM generated answer.
            - 'source_documents': List of retrieved Document objects.
    """
    try:
        fetch_k = top_k * 2 if use_reranking else top_k
        retriever = get_retriever(vector_store, top_k=fetch_k)
        docs = retriever.invoke(question)
        if use_reranking:
            from reranker import rerank_documents
            docs = rerank_documents(question, docs, top_n=top_k)
        docs = _enrich_metadata_docs(question, vector_store, docs)
        
        # Format the context for prompt
        context_parts = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source', 'Unknown')
            src_name = src.split('\\')[-1].split('/')[-1]
            page_num = doc.metadata.get('page', 0) + 1
            context_parts.append(f"[Document {i+1}] {src_name} - Page {page_num}:\n{doc.page_content}")
            
        context_str = "\n\n".join(context_parts)
        
        qa_prompt = get_qa_prompt()
        formatted_messages = qa_prompt.format_messages(context=context_str, question=question)
        
        llm = get_llm(model_name=llm_model)
        response = llm.invoke(formatted_messages)
        
        return {
            "answer": response.content,
            "source_documents": docs
        }
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise e

def query_rag_pipeline_stream(
    question: str,
    vector_store: Chroma,
    top_k: int = 4,
    llm_model: str = config.DEFAULT_LLM_MODEL,
    use_reranking: bool = False
) -> Generator[Dict[str, Any], None, None]:
    """Execute the RAG pipeline and yield results as a stream.

    Args:
        question (str): User question.
        vector_store (Chroma): The Chroma vector store.
        top_k (int): Number of sources to retrieve.
        llm_model (str): LLM model name.
        use_reranking (bool): Whether to use CrossEncoder re-ranking.

    Yields:
        Dict[str, Any]: A dictionary containing either 'source_documents' (first yield)
                         or 'answer_chunk' (subsequent token chunks).
    """
    try:
        # 1. Retrieve sources
        fetch_k = top_k * 2 if use_reranking else top_k
        retriever = get_retriever(vector_store, top_k=fetch_k)
        docs = retriever.invoke(question)
        if use_reranking:
            from reranker import rerank_documents
            docs = rerank_documents(question, docs, top_n=top_k)
        docs = _enrich_metadata_docs(question, vector_store, docs)
        
        # Yield the source documents immediately so the UI can capture and display them
        yield {"source_documents": docs}
        
        # 2. Setup prompt and context
        context_parts = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source', 'Unknown')
            src_name = src.split('\\')[-1].split('/')[-1]
            page_num = doc.metadata.get('page', 0) + 1
            context_parts.append(f"[Document {i+1}] {src_name} - Page {page_num}:\n{doc.page_content}")
            
        context_str = "\n\n".join(context_parts)
        
        qa_prompt = get_qa_prompt()
        formatted_messages = qa_prompt.format_messages(context=context_str, question=question)
        
        # 3. Stream the LLM response
        llm = get_llm(model_name=llm_model)
        for chunk in llm.stream(formatted_messages):
            yield {"answer_chunk": chunk.content}
            
    except Exception as e:
        logger.error(f"Error in streaming RAG pipeline: {str(e)}")
        raise e

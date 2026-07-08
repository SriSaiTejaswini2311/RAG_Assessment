import os
import sys
import logging
from pdf_loader import load_pdf
from text_splitter import split_documents
from embeddings import get_embeddings_model
from vector_store import initialize_vector_store, add_documents, clear_vector_store
from retriever import get_retriever
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("=== STARTING RAG DRY-RUN VERIFICATION ===")
    
    # 1. Load sample PDF
    pdf_path = "sample.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} does not exist.")
        sys.exit(1)
        
    print(f"\n1. Loading PDF: {pdf_path}")
    docs = load_pdf(pdf_path)
    print(f"Loaded {len(docs)} pages.")
    for i, doc in enumerate(docs):
        print(f"  Page {i+1} character count: {len(doc.page_content)}")
        
    # 2. Split documents
    print(f"\n2. Splitting documents (chunk_size={config.DEFAULT_CHUNK_SIZE}, overlap={config.DEFAULT_CHUNK_OVERLAP})")
    chunks = split_documents(docs, chunk_size=config.DEFAULT_CHUNK_SIZE, chunk_overlap=config.DEFAULT_CHUNK_OVERLAP)
    print(f"Generated {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i+1} (Source: {chunk.metadata.get('source')}, Page: {chunk.metadata.get('page') + 1}):")
        print(f"    Content preview: {chunk.page_content[:150].strip()}...")
        
    # 3. Embedding model
    print(f"\n3. Initializing Embedding Model: {config.DEFAULT_EMBEDDING_MODEL}")
    embeddings = get_embeddings_model(config.SUPPORTED_EMBEDDING_MODELS[config.DEFAULT_EMBEDDING_MODEL])
    print("Embedding model initialized.")
    
    # 4. Vector Store
    print("\n4. Initializing Vector Store (ChromaDB)")
    vector_store = initialize_vector_store(embeddings)
    
    # Clear store first to start clean
    print("Clearing vector store...")
    clear_vector_store(vector_store)
    vector_store = initialize_vector_store(embeddings) # Reinitialize
    
    # Add documents
    print("Adding document chunks to vector store...")
    add_documents(vector_store, chunks)
    print("Chunks added.")
    
    # 5. Semantic Search
    print("\n5. Testing Retriever / Semantic Search")
    retriever = get_retriever(vector_store, top_k=2)
    query = "Who founded Acme Corporation and when?"
    print(f"Query: '{query}'")
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} relevant documents.")
    for i, doc in enumerate(retrieved_docs):
        print(f"  Result {i+1} (Page {doc.metadata.get('page') + 1}):")
        print(f"    Content: {doc.page_content.strip()}")
        
    # 6. LLM Check
    print("\n6. Groq API Key Check")
    if config.is_groq_api_key_set():
        print("GROQ_API_KEY is configured. Running end-to-end QA pipeline...")
        from rag import query_rag_pipeline
        response = query_rag_pipeline(query, vector_store, top_k=2)
        print("\n--- LLM Response ---")
        print(response['answer'])
        print("--------------------")
    else:
        print("GROQ_API_KEY is not set. Skipping LLM query call. Pipeline is structurally functional!")
        
    print("\n=== VERIFICATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()

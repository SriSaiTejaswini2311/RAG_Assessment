# Production-Quality PDF QA RAG Application

An production-grade Retrieval-Augmented Generation (RAG) system built in Python to perform grounded Question Answering over PDF documents. The application is designed to be clean, modular, and interview-ready.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [How RAG Works](#-how-rag-works)
5. [Folder Structure](#-folder-structure)
6. [Installation & Setup](#-installation--setup)
7. [Environment Variables](#-environment-variables)
8. [Sample Questions for Testing](#-sample-questions-for-testing)
9. [Technologies Used](#-technologies-used)
10. [Future Improvements](#-future-improvements)

---

## 🔍 Project Overview
This application parses one or more PDF files, creates structured text chunks, runs them through a HuggingFace sentence transformer to compute dense vector embeddings, and stores them in a local ChromaDB database. When users submit questions, the retriever queries ChromaDB for the Top-K relevant text passages and sends them as context to a Groq LLM (e.g., `llama-3.3-70b-versatile`) under strict prompt filters.

**Strict Grounding Rule**: The LLM will only answer using facts explicitly mentioned in the context. If the answer is not present, it will output:
`"I couldn't find that information in the uploaded documents."`

---

## ✨ Key Features
- **Multiple PDF Support**: Upload and index multiple PDF files concurrently.
- **Configurable RAG Parameters**: Interactive settings in the UI sidebar for chunk size, chunk overlap, and Top-K document retrieval.
- **Configurable Embedding Models**: Support for `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`.
- **Streaming Responses**: Clean token-by-token text streaming in the chat interface.
- **Source Citations**: Renders interactive expanders detailing filenames and 1-indexed page numbers matching the retrieved content.
- **Database Control**: Clear and rebuild the vector database index directly via the UI.
- **Chat Management**: Clear the conversational history thread without deleting the document index.
- **Graceful Fallbacks**: Paste your Groq API key directly into the UI if the env variable is missing.

---

## 📐 System Architecture

The application uses a **two-stage retrieval architecture** to optimize relevance:

```mermaid
flowchart TD
    User([User Query]) --> RetrieverModule[retriever.py: Similarity Retriever]
    
    subgraph Document Indexing Pipeline
        PDF[PDF Files] --> Loader[pdf_loader.py: PyPDFLoader]
        Loader --> Splitter[text_splitter.py: RecursiveCharacterSplitter]
        Splitter --> Embedder[embeddings.py: HuggingFaceEmbeddings]
        Embedder --> VectorStore[(vector_store.py: ChromaDB)]
    end
    
    VectorStore --> RetrieverModule
    RetrieverModule -->|Retrieve top_k * 2 candidates| RerankerModule[reranker.py: Cross-Encoder]
    RerankerModule -->|Score & Select Top-K Chunks| PromptEngine[prompts.py: System Prompt]
    PromptEngine -->|Formatted Messages| GroqLLM[rag.py: ChatGroq]
    GroqLLM -->|Stream Answer & Citations| UI[app.py: Streamlit]
    UI --> User
```

---

## 🛠️ How RAG Works
1. **Document Loading**: PyPDFLoader reads the PDF binary page-by-page, converting layout pages into document objects.
2. **Text Chunking**: Text is split recursively based on standard paragraph, sentence, and character boundaries (`\n\n`, `\n`, ` `, `""`) to keep coherent blocks of size $N$ characters with a slide overlap to maintain semantic continuity between boundary splits.
3. **Dense Vector Embeddings**: Text chunks are passed through a sentence-transformer model that maps words and sentences to dense floating-point vector spaces capturing semantic meaning.
4. **Vector Database**: ChromaDB stores the vectors and indexes them using Hierarchical Navigable Small World (HNSW) graphs.
5. **Semantic Retrieval**: Queries are converted into the same embedding space, and vector cosine similarity is computed. The top $K$ nearest vector chunks are retrieved.
6. **LLM Context Synthesis**: The LLM compiles the context alongside the query. It synthesizes a grounded answer, completely bounded from fabricating outside details.

---

## 📂 Folder Structure
```text
rag-project/
│
├── data/                    # Temporary folder storing uploaded PDF assets
├── chroma_db/               # Persistent SQLite-backed database files for Chroma
│
├── app.py                   # Streamlit web-based UI code
├── rag.py                   # Orchestration module binding LLM, retriever, and streaming
├── config.py                # Environment parser and static defaults config
│
├── pdf_loader.py            # PyPDF text extraction module
├── text_splitter.py         # Recursive text splitting wrapper with custom chunk_id
├── embeddings.py            # HuggingFace Embeddings wrapper
├── vector_store.py          # Chroma database manager (Init, Add, Clear)
├── retriever.py             # Similarity search configuration
├── reranker.py              # Cross-Encoder (ms-marco-MiniLM) Stage-2 Reranking
├── prompts.py               # Prompt templates & QA instructions
│
├── requirements.txt         # Project package dependencies
├── .env.example             # Template file for API keys
├── README.md                # System documentation
│
├── generate_sample_pdf.py   # Script to build sample.pdf
├── sample.pdf               # Mock corporate report containing testing facts
└── test_rag.py              # Diagnostic dry-run script
```

---

## 🚀 Installation & Setup

1. **Clone the project workspace** and navigate to the folder:
   ```bash
   cd Assessment/RAG
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows (cmd/powershell):
   venv\Scripts\activate
   # On MacOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the testing sample PDF**:
   ```bash
   python generate_sample_pdf.py
   ```

5. **Configure environment keys**:
   Create a `.env` file from the example:
   ```bash
   copy .env.example .env
   ```
   Add your Groq API Key:
   ```text
   GROQ_API_KEY=gsk_your_groq_api_key_goes_here
   ```

6. **Run the diagnostic pipeline**:
   Ensure the retrieval works correctly before starting the web UI:
   ```bash
   python test_rag.py
   ```

7. **Launch the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 🔑 Environment Variables
| Key | Type | Description | Required |
|-----|------|-------------|----------|
| `GROQ_API_KEY` | String | API key to run queries through ChatGroq. | Yes (or via UI fallback) |

---

## 💡 Sample Queries and Outputs

Here are actual queries run against indexed documents demonstrating extraction, synthesis, and grounding validation:

### 📄 Test Case 1: Legal Metadata Questions (`DP Joshi.pdf` indexed)

*   **Query**: `"Who delivered the majority judgment?"`
    *   **Answer**: *"The majority judgment was delivered by T.L. Venkatarama Ayyar, J."*
    *   **Citations**: `DP Joshi.pdf` (Page 1)
*   **Query**: `"When was the judgment delivered?"`
    *   **Answer**: *"The judgment was delivered on January 27, 1955."*
    *   **Citations**: `DP Joshi.pdf` (Page 1)

### 📄 Test Case 2: Fact Extraction (`sample.pdf` indexed)

*   **Query**: `"Who founded Acme Corporation and when?"`
    *   **Answer**: *"Acme Corporation was founded by Jane Doe in 2028."*
    *   **Citations**: `sample.pdf` (Page 1)
*   **Query**: `"What is the retail price of the Anti-Gravity Boots v4.2?"`
    *   **Answer**: *"The retail price of the Anti-Gravity Boots v4.2 is $12,499 per pair."*
    *   **Citations**: `sample.pdf` (Page 2)

### 📄 Test Case 3: Out-of-Domain Grounding (Hallucination Prevention)

*   **Query**: `"What is the capital of Japan?"`
    *   **Answer**: *"I couldn't find that information in the uploaded documents."*
    *   **Explanation**: The system prompt actively prevents the LLM from using pre-trained external knowledge, verifying strict compliance.

---

## 🛠️ Technologies Used
- **Streamlit**: Web frontend and interactive sidebar.
- **LangChain**: Application framework, Prompting, and LCEL abstractions.
- **ChatGroq**: High-speed inference using LLaMA models.
- **HuggingFace Embeddings**: Local text vector representation.
- **ChromaDB**: SQLite-backed local vector store.
- **ReportLab**: Programmatic PDF compilation.
- **PyPDF**: PDF text scanning.

---

## 🚀 Future Improvements
1. **Hybrid Retrieval**: Combine dense semantic embeddings with sparse keyword search (BM25) to improve acronym matching.
2. **Metadata Filtering**: Support filtering documents by file names, sizes, or dates directly in the retrieval query.
3. **Chunk Summary Pre-indexing**: Store parent-document summaries alongside chunks to improve macro-level context matching.
4. **Conversation History Memory**: Implement conversational context retrieval by compressing preceding dialogue inside a contextualizer chain.

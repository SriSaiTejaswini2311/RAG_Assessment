import streamlit as st
import os
import shutil
import logging
from pathlib import Path

import config
from pdf_loader import load_multiple_pdfs
from text_splitter import split_documents
from embeddings import get_embeddings_model
from vector_store import initialize_vector_store, add_documents, clear_vector_store
import importlib
import rag
importlib.reload(rag)
from rag import query_rag_pipeline_stream

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG PDF QA Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for UI visual appeal
st.markdown("""
<style>
    /* Import modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Apply clean font globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* App background subtle radial gradient */
    .stApp {
        background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.02), transparent 45%),
                    radial-gradient(circle at bottom left, rgba(29, 78, 216, 0.02), transparent 45%);
    }

    /* Glowing header title with Jakarta font scale */
    .main-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.12rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Document list item cards with hover motion */
    .doc-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 0.8rem;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color, rgba(0,0,0,0.04));
        border-left: 4px solid var(--primary-color);
        border-radius: 10px;
        margin-bottom: 0.6rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.01);
    }
    .doc-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        border-left-color: #1d4ed8;
    }
    
    /* Premium overrides for all Streamlit buttons */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    
    /* Primary buttons (Index / Process) hover effects */
    div.stButton > button[type="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.22) !important;
    }
    
    /* Secondary buttons (Clear DB / Chat) hover effects */
    div.stButton > button[type="secondary"]:hover {
        transform: translateY(-1px) !important;
        background-color: var(--secondary-background-color) !important;
    }

    /* Slick Chat message bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 1.1rem 1.3rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid var(--border-color, rgba(0,0,0,0.03)) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.01) !important;
        transition: transform 0.2s ease !important;
    }
    
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.02) !important;
    }

    /* Expander card custom headers */
    .streamlit-expanderHeader {
        background-color: var(--secondary-background-color) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border-color, rgba(0,0,0,0.04)) !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = config.DEFAULT_EMBEDDING_MODEL
if "top_k" not in st.session_state:
    st.session_state.top_k = config.DEFAULT_TOP_K
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = config.DEFAULT_CHUNK_SIZE
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = config.DEFAULT_CHUNK_OVERLAP
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "use_reranking" not in st.session_state:
    st.session_state.use_reranking = False
if "llm_model" not in st.session_state:
    st.session_state.llm_model = config.DEFAULT_LLM_MODEL

@st.cache_resource
def load_embeddings(model_name: str):
    """Cached helper to load and memoize the embedding model."""
    hf_model_path = config.SUPPORTED_EMBEDDING_MODELS[model_name]
    return get_embeddings_model(hf_model_path)

# Try loading the vector store automatically on startup if files exist on disk
if st.session_state.vector_store is None:
    try:
        embeddings = load_embeddings(st.session_state.embedding_model)
        st.session_state.vector_store = initialize_vector_store(embeddings)
        stored_data = st.session_state.vector_store.get()
        if stored_data and stored_data.get("metadatas"):
            sources = set()
            for meta in stored_data["metadatas"]:
                if meta and "source" in meta:
                    src_name = meta["source"].split('\\')[-1].split('/')[-1]
                    sources.add(src_name)
            st.session_state.processed_files = list(sources)
    except Exception as e:
        logger.warning(f"Could not load pre-existing vector store: {str(e)}")

# Sidebar Configuration Control Panel
with st.sidebar:
    # Render custom high-tech logo
    logo_path = Path(__file__).parent / "rag_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.title("📚 RAG Panel")
    
    st.markdown("---")
    
    # 0. API Key Configuration
    st.subheader("🔑 API Key")
    env_key_set = bool(os.getenv("GROQ_API_KEY", "").strip())
    placeholder_text = "Loaded from .env" if env_key_set else "gsk_..."
    
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder=placeholder_text,
        help="Paste your custom Groq API key here. If left blank, the app will try to load it from the .env file."
    )
    
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
        config.GROQ_API_KEY = api_key_input
    elif env_key_set:
        config.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        st.session_state.groq_api_key = ""
        
    st.markdown("---")
    
    # 1. Document Upload
    st.subheader("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDFs to build the local index."
    )
    
    # 2. Advanced Parameters
    with st.expander("⚙️ Parameters", expanded=False):
        # Groq LLM Model selection
        GROQ_MODELS = {
            "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Best quality)",
            "llama-3.1-8b-instant": "Llama 3.1 8B Instant (Fastest)",
            "mixtral-8x7b-32768": "Mixtral 8x7B (Long context)",
            "gemma2-9b-it": "Gemma 2 9B (Lightweight)",
        }
        st.session_state.llm_model = st.selectbox(
            "🤖 Groq LLM Model",
            options=list(GROQ_MODELS.keys()),
            index=list(GROQ_MODELS.keys()).index(
                st.session_state.llm_model
                if st.session_state.llm_model in GROQ_MODELS
                else config.DEFAULT_LLM_MODEL
            ),
            format_func=lambda x: GROQ_MODELS[x],
            help="Select the Groq model for generating answers. Switch models if you hit daily rate limits."
        )

        st.markdown("---")

        # Embedding Model selection
        prev_model = st.session_state.embedding_model
        selected_model = st.selectbox(
            "🧠 Embedding Model",
            options=list(config.SUPPORTED_EMBEDDING_MODELS.keys()),
            index=list(config.SUPPORTED_EMBEDDING_MODELS.keys()).index(st.session_state.embedding_model),
            help="Changing the model requires database wipe."
        )
        if selected_model != prev_model:
            st.session_state.embedding_model = selected_model
            st.warning("⚠️ Embedding model changed! Clear Database and reprocess PDFs.")
            
        # Top-K Retrieval
        st.session_state.top_k = st.slider(
            "Top-K retrieved chunks",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of text passages to retrieve."
        )
        
        # Re-ranking toggle
        st.session_state.use_reranking = st.checkbox(
            "🔥 Enable Re-ranking (CrossEncoder)",
            value=st.session_state.use_reranking,
            help="Use a second-stage local Cross-Encoder model to re-order the retrieved chunks for maximum precision."
        )
        
        # Chunk sizes
        st.session_state.chunk_size = st.number_input(
            "Chunk Size (chars)",
            min_value=100,
            max_value=5000,
            value=st.session_state.chunk_size,
            step=100
        )
        st.session_state.chunk_overlap = st.number_input(
            "Chunk Overlap (chars)",
            min_value=0,
            max_value=1000,
            value=st.session_state.chunk_overlap,
            step=50
        )
        
    st.markdown("---")
    
    # 3. Action Button
    process_clicked = st.button(
        "🚀 Process & Index",
        use_container_width=True,
        type="primary",
        disabled=not uploaded_files
    )
    
    if process_clicked and uploaded_files:
        # Recreate uploads folder
        if config.DATA_DIR.exists():
            shutil.rmtree(config.DATA_DIR)
        config.DATA_DIR.mkdir(exist_ok=True, parents=True)
        
        file_paths = []
        for file in uploaded_files:
            dest_path = config.DATA_DIR / file.name
            with open(dest_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(str(dest_path))
            
        with st.status("Indexing documents... Please wait", expanded=True) as status:
            try:
                st.write("📖 Reading uploaded PDF pages...")
                documents = load_multiple_pdfs(file_paths)
                
                st.write("✂️ Splitting pages into text chunks...")
                chunks = split_documents(
                    documents, 
                    chunk_size=st.session_state.chunk_size, 
                    chunk_overlap=st.session_state.chunk_overlap
                )
                
                st.write("🧠 Instantiating embedding model...")
                embeddings = load_embeddings(st.session_state.embedding_model)
                
                st.write("🗄️ Loading vector database...")
                st.session_state.vector_store = initialize_vector_store(embeddings)
                
                st.write("🧹 Cleaning old indexes...")
                clear_vector_store(st.session_state.vector_store)
                st.session_state.vector_store = initialize_vector_store(embeddings)
                
                st.write("📥 Inserting chunks into ChromaDB...")
                add_documents(st.session_state.vector_store, chunks)
                
                status.update(label="✅ Documents indexed successfully!", state="complete", expanded=False)
                
                st.session_state.processed_files = [file.name for file in uploaded_files]
                st.success(f"Processed {len(uploaded_files)} PDF(s) ({len(chunks)} chunks).")
            except Exception as e:
                status.update(label="❌ Indexing failed!", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
                
    st.markdown("---")
    
    # 4. List of uploaded files
    st.subheader("📚 Indexed PDFs")
    if st.session_state.processed_files:
        for fname in st.session_state.processed_files:
            st.markdown(f"""
            <div class="doc-item">
                <span style="font-size: 1.1rem; margin-right: 0.5rem;">📄</span>
                <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.85rem;">
                    <b>{fname}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No documents indexed. Upload PDFs above.")
        
    st.markdown("---")
    
    # 5. Reset button
    clear_db_clicked = st.button(
        "🗑️ Clear Database",
        use_container_width=True,
        type="secondary"
    )
    if clear_db_clicked:
        if st.session_state.vector_store is not None:
            try:
                clear_vector_store(st.session_state.vector_store)
                st.session_state.vector_store = None
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                if config.DATA_DIR.exists():
                    shutil.rmtree(config.DATA_DIR)
                    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
                st.success("Vector database reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting database: {str(e)}")
        else:
            st.info("Vector database is already empty.")

# Main Interface Area
col1, col2 = st.columns([1, 7])
with col1:
    logo_path = Path(__file__).parent / "rag_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=75)
with col2:
    st.markdown('<div class="main-header">Research Assistant RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle" style="margin-bottom: 0rem;">Upload research documents, textbooks, or manuals, and ask questions. Supported by grounding filters.</div>', unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# Instructions Banner
st.info(
    "🔒 **Strict Grounding Policy:** The model answers queries using ONLY the retrieved text chunks from the PDF. "
    "If the text chunks do not contain the answer, the model responds with: "
    '"I couldn\'t find that information in the uploaded documents."'
)

# Verify API Keys
if not config.is_groq_api_key_set():
    st.error("🔑 **Groq API Key Missing**")
    st.info("Please enter your Groq API Key in the sidebar control panel to start using the app.")
    st.stop()

# Validate DB State
db_ready = False
if st.session_state.vector_store is not None:
    try:
        count = len(st.session_state.vector_store.get()["ids"])
        db_ready = count > 0
    except Exception:
        db_ready = False

# Layout header columns for clear buttons
chat_col, clear_col = st.columns([6, 1])
with clear_col:
    clear_chat_clicked = st.button("🧼 Clear Chat", use_container_width=True)
    if clear_chat_clicked:
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    avatar_char = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_char):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources & Citations", expanded=False):
                # Deduplicate source file/pages
                dedup = {}
                for src in msg["sources"]:
                    name = src["name"]
                    p = src["page"]
                    if name not in dedup:
                        dedup[name] = set()
                    dedup[name].add(p)
                
                for name, pages in dedup.items():
                    pages_str = ", ".join([str(p) for p in sorted(pages)])
                    st.markdown(f"- **{name}** (Page(s): {pages_str})")
                
                # Render source text snippets
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Snippet {idx+1}: {src['name']} (Page {src['page']})**")
                    st.info(src["content"])

# User Chat Prompt Interface
if not db_ready:
    st.info("👋 Welcome! Please upload PDF documents in the sidebar and click **Process & Index** to begin chatting.")
else:
    user_query = st.chat_input("Ask a question about your documents...")
    if user_query:
        # Display user query
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display assistant streaming answer
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            stream_generator = query_rag_pipeline_stream(
                question=user_query,
                vector_store=st.session_state.vector_store,
                top_k=st.session_state.top_k,
                llm_model=st.session_state.llm_model,
                use_reranking=st.session_state.use_reranking
            )
            
            full_response = ""
            source_docs = []
            
            with st.spinner("Retrieving facts and generating response..."):
                try:
                    for val in stream_generator:
                        if "source_documents" in val:
                            source_docs = val["source_documents"]
                        elif "answer_chunk" in val:
                            full_response += val["answer_chunk"]
                            response_placeholder.markdown(full_response + "▌")
                            
                    response_placeholder.markdown(full_response)
                    
                    if source_docs:
                        with sources_placeholder.container():
                            with st.expander("📚 View Sources & Citations", expanded=False):
                                dedup = {}
                                for doc in source_docs:
                                    src_name = doc.metadata.get('source', 'Unknown').split('\\')[-1].split('/')[-1]
                                    page_num = doc.metadata.get('page', 0) + 1
                                    if src_name not in dedup:
                                        dedup[src_name] = set()
                                    dedup[src_name].add(page_num)
                                    
                                for name, pages in dedup.items():
                                    pages_str = ", ".join([str(p) for p in sorted(pages)])
                                    st.markdown(f"- **{name}** (Page(s): {pages_str})")
                                    
                                # Detailed snippets
                                for idx, doc in enumerate(source_docs):
                                    sname = doc.metadata.get('source', 'Unknown').split('\\')[-1].split('/')[-1]
                                    pnum = doc.metadata.get('page', 0) + 1
                                    st.markdown(f"**Snippet {idx+1}: {sname} (Page {pnum})**")
                                    st.info(doc.page_content)
                                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": [{"name": doc.metadata.get('source', 'Unknown').split('\\')[-1].split('/')[-1], "page": doc.metadata.get('page', 0) + 1, "content": doc.page_content} for doc in source_docs]
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    # Remove the question from history if it failed
                    st.session_state.chat_history.pop()

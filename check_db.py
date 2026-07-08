from langchain_chroma import Chroma
from embeddings import get_embeddings_model
import config

embeddings = get_embeddings_model(config.SUPPORTED_EMBEDDING_MODELS[config.DEFAULT_EMBEDDING_MODEL])
vector_store = Chroma(
    collection_name="pdf_qa_collection",
    embedding_function=embeddings,
    persist_directory=str(config.CHROMA_DB_DIR)
)

data = vector_store.get(where={"page": 0})
print("Documents list:", data.get("documents"))
print("Metadatas list:", data.get("metadatas"))
print("IDs list:", data.get("ids"))

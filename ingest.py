from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
import os

def run_ingestion(docs_dir="docs", db_path="./chroma_db", collection_name="my_notes"):
    """
    Ingests documents from docs_dir into a Chroma collection named collection_name at db_path.
    Recreates the collection to avoid duplicate node entries.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Created empty '{docs_dir}' directory.")
        return 0

    print(f"Loading documents from /{docs_dir} ...")
    documents = SimpleDirectoryReader(docs_dir, required_exts=[".pdf", ".md", ".txt"]).load_data()
    print(f"Loaded {len(documents)} document chunks")
    
    if not documents:
        print("No documents found to index.")
        return 0

    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Recreate the collection to prevent duplicate entries
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Cleared existing collection '{collection_name}' to prevent duplicates.")
    except Exception:
        # Collection might not exist yet
        pass
        
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Embedding documents (this may take a minute)...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print(f"Done! Your docs are stored in {db_path}")
    return len(documents)

if __name__ == "__main__":
    run_ingestion()
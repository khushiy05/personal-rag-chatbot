from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("📄 Loading documents from /docs ...")
documents = SimpleDirectoryReader("docs", required_exts=[".pdf", ".md", ".txt"]).load_data()
print(f"✅ Loaded {len(documents)} document chunks")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_notes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("🔄 Embedding documents (this may take a minute)...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
print("✅ Done! Your docs are stored in ./chroma_db")
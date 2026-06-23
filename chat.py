from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_notes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

query_engine = index.as_query_engine(similarity_top_k=3)

print("🤖 RAG Chatbot ready! Ask anything about your notes.")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ["quit", "exit"]:
        break
    if not question:
        continue

    response = query_engine.query(question)
    print(f"\n🤖 Answer: {response}\n")

    print("📚 Sources used:")
    for node in response.source_nodes:
        fname = node.metadata.get("file_name", "unknown")
        score = round(node.score, 3) if node.score else "N/A"
        print(f"  • {fname} (relevance: {score})")
    print()
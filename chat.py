from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Using llama3.2 (3B) since it's smarter, falls back gracefully if config changes
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_notes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    similarity_top_k=3,
    system_prompt="You are Khushi's friendly AI assistant. Be warm and answer questions using context documents. If the user asks general or greeting questions, be helpful and reply directly."
)

print("RAG Chatbot ready! Ask anything about your notes (with conversation memory).")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ["quit", "exit"]:
        break
    if not question:
        continue

    print("\nAnswer: ", end="", flush=True)
    response_stream = chat_engine.stream_chat(question)
    
    for token in response_stream.response_gen:
        print(token, end="", flush=True)
    print("\n")

    if response_stream.source_nodes:
        print("Sources used:")
        seen_sources = set()
        for node in response_stream.source_nodes:
            fname = node.metadata.get("file_name", "unknown")
            score = round(node.score, 3) if node.score else "N/A"
            if fname not in seen_sources:
                print(f"  - {fname} (relevance: {score})")
                seen_sources.add(fname)
        print()
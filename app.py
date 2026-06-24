import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

st.set_page_config(page_title="Khushi's AI Assistant", page_icon="✦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0d0d0f !important;
    color: #e8e6e1 !important;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

.top-header { text-align: center; padding: 2rem 0 0.5rem 0; }
.logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.subtitle { color: #6b7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #1a1a1f; border: 1px solid #2a2a32;
    padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; color: #6b7280; margin: 0.8rem 0 1.5rem 0;
}
.dot { width:6px; height:6px; background:#34d399; border-radius:50%; display:inline-block; }

[data-testid="stChatMessage"] { background: transparent !important; }
[data-testid="stChatMessageContent"] { font-size: 0.93rem; line-height: 1.6; }

[data-testid="stChatInput"] textarea {
    background: #1a1a1f !important;
    border: 1px solid #2a2a32 !important;
    border-radius: 14px !important;
    color: #e8e6e1 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div class="logo">✦ Khushi's AI Assistant</div>
    <div class="subtitle">Powered by your personal documents</div>
</div>
<div style="text-align:center">
    <div class="badge"><span class="dot"></span> Running locally · 100% private</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_index():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Ollama(model="llama3.2:1b", request_timeout=180.0)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("my_notes")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return index.as_query_engine(similarity_top_k=3)

with st.spinner("Loading your knowledge base..."):
    query_engine = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:2rem; color:#4b5563;">
        <div style="font-size:2rem; margin-bottom:0.8rem;">🗂️</div>
        <div style="font-size:1rem; color:#6b7280;">Ask me anything about your documents or just say hi!</div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📚 Sources: " + " · ".join(msg["sources"]))

if prompt := st.chat_input("Say hi or ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    casual_keywords = [
        "hi", "hello", "hey", "how are you", "what are you", "who are you",
        "thanks", "thank you", "bye", "good morning", "good evening",
        "good night", "what can you do", "help", "nice", "cool", "ok", "okay"
    ]
    is_casual = any(word in prompt.lower() for word in casual_keywords)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_casual:
                llm = Ollama(model="llama3.2:1b", request_timeout=120.0)
                answer = str(llm.complete(
                    f"You are Khushi's friendly AI assistant. Be warm and short like ChatGPT.\n\nUser: {prompt}\nAssistant:"
                ))
                sources = []
            else:
                response = query_engine.query(prompt)
                answer = str(response)
                sources = list({
                    node.metadata.get("file_name", "unknown")
                    for node in response.source_nodes
                })

        st.markdown(answer)
        if sources:
            st.caption("📚 Sources: " + " · ".join(sources))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
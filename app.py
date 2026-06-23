import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Khushi's AI Assistant",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

/* Root reset */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0f !important;
    color: #e8e6e1 !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stMain"] {
    background: #0d0d0f !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Top header */
.top-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.top-header .logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.top-header .subtitle {
    color: #6b7280;
    font-size: 0.85rem;
    font-weight: 400;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Chat container */
.chat-wrapper {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* Message bubbles */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.8rem 0;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: #fff;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 2px 12px rgba(124,58,237,0.25);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 0.8rem 0;
    gap: 0.6rem;
    align-items: flex-start;
}
.msg-bot .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-bot .bubble {
    background: #1a1a1f;
    border: 1px solid #2a2a32;
    color: #e8e6e1;
    padding: 0.75rem 1.1rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
}
.msg-bot .sources {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #2a2a32;
}
.msg-bot .sources span {
    background: #252530;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    margin-right: 0.3rem;
    display: inline-block;
    margin-top: 0.2rem;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #4b5563;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 1rem; }
.empty-state h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    color: #6b7280;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.empty-state .suggestions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 1.2rem;
}
.suggestion-pill {
    background: #1a1a1f;
    border: 1px solid #2a2a32;
    color: #9ca3af;
    padding: 0.45rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
}

/* Input area */
[data-testid="stChatInput"] {
    background: #1a1a1f !important;
    border: 1px solid #2a2a32 !important;
    border-radius: 14px !important;
    color: #e8e6e1 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a32; border-radius: 4px; }

/* Spinner */
[data-testid="stSpinner"] { color: #7c3aed !important; }

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #1a1a1f;
    border: 1px solid #2a2a32;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 1.5rem;
}
.status-dot {
    width: 6px; height: 6px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ── Load index (cached) ───────────────────────────────────────────────────────
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


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <div class="logo">✦ Khushi's AI Assistant</div>
    <div class="subtitle">Powered by your personal documents</div>
</div>
<div style="text-align:center">
    <div class="status-badge">
        <div class="status-dot"></div>
        Running locally · 100% private
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading your knowledge base..."):
    query_engine = load_index()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🗂️</div>
        <h3>Ask me anything about your documents</h3>
        <p style="font-size:0.82rem">I've read all your files and I'm ready to help.</p>
        <div class="suggestions">
            <div class="suggestion-pill">What's in my resume?</div>
            <div class="suggestion-pill">Summarize my notes</div>
            <div class="suggestion-pill">What are my skills?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="bubble">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                pills = "".join(f"<span>📄 {s}</span>" for s in msg["sources"])
                sources_html = f'<div class="sources">Sources: {pills}</div>'
            st.markdown(f"""
            <div class="msg-bot">
                <div class="avatar">✦</div>
                <div class="bubble">
                    {msg["content"]}
                    {sources_html}
                </div>
            </div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = query_engine.query(prompt)
        answer = str(response)
        sources = list({
            node.metadata.get("file_name", "unknown")
            for node in response.source_nodes
        })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()
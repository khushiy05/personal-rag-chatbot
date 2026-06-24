import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import chromadb
import os
import urllib.request
import json
from ingest import run_ingestion

# Page configurations
st.set_page_config(page_title="Personal AI Assistant", page_icon="💬", layout="wide")

# Minimalist, clean Light Theme CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main Page Layout */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    font-family: 'Inter', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #f9fafb !important;
    border-right: 1px solid #e5e7eb !important;
}

/* Sidebar Text & Labels Override */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h3 {
    color: #374151 !important;
}

/* Hide Streamlit default headers & footers */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

/* Main Header Styling */
.main-title {
    font-size: 1.8rem;
    font-weight: 600;
    text-align: center;
    margin-top: 1.5rem;
    color: #1f2937;
}
.main-subtitle {
    font-size: 0.9rem;
    text-align: center;
    color: #6b7280;
    margin-bottom: 2rem;
}

/* Chat Input Bar */
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 24px !important;
    color: #1f2937 !important;
    padding-left: 1.2rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
}

[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    padding-bottom: 2rem !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border-bottom: 1px solid #f3f4f6 !important;
    padding: 1.5rem 2rem !important;
}

/* Force dark readable text */
[data-testid="stChatMessageContent"], 
[data-testid="stChatMessageContent"] p, 
[data-testid="stChatMessageContent"] li, 
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] div,
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 {
    color: #1f2937 !important;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Source Card Styling */
.source-box {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.8rem;
    margin-top: 0.5rem;
}
.source-meta {
    font-weight: 600;
    color: #0f766e;
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
}
.source-text {
    font-size: 0.85rem;
    color: #4b5563;
    line-height: 1.4;
    background-color: #ffffff;
    padding: 0.4rem;
    border-radius: 4px;
    border: 1px solid #f3f4f6;
}
</style>
""", unsafe_allow_html=True)

# 1. Global Setup (Must execute on every rerun to prevent default model timeouts)
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.embed_model = load_embed_model()

# Helper function to discover available local Ollama models
def get_installed_ollama_models():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode())
            models = [model["name"] for model in data.get("models", [])]
            clean_models = []
            for m in models:
                if m.endswith(":latest"):
                    base = m[:-7]
                    if base not in clean_models:
                        clean_models.append(base)
                else:
                    clean_models.append(m)
            return clean_models if clean_models else ["llama3.2"]
    except Exception:
        return ["llama3.2", "llama3.2:1b"]

# Initialize state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_version" not in st.session_state:
    st.session_state.index_version = 0
if "current_model" not in st.session_state:
    st.session_state.current_model = ""
if "current_top_k" not in st.session_state:
    st.session_state.current_top_k = 3
if "current_temp" not in st.session_state:
    st.session_state.current_temp = 0.3
if "current_sys_prompt" not in st.session_state:
    st.session_state.current_sys_prompt = ""
if "current_index_version" not in st.session_state:
    st.session_state.current_index_version = -1
if "current_chat_target" not in st.session_state:
    st.session_state.current_chat_target = "All Documents"

# Load cached vector index
@st.cache_resource(show_spinner=False)
def load_cached_index(index_version):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("my_notes")
    db_count = chroma_collection.count()
    
    if db_count == 0:
        return None, 0
        
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return index, db_count

# Read uploaded docs
if not os.path.exists("docs"):
    os.makedirs("docs")
existing_docs = os.listdir("docs")

# --- Left Sidebar Layout ---
st.sidebar.markdown("<div style='padding-top: 1rem;'></div>", unsafe_allow_html=True)

# 1. Clear Chat History
if st.sidebar.button("🧹 Clear Chat", use_container_width=True):
    st.session_state.messages = []
    if "chat_engine" in st.session_state:
        st.session_state.chat_engine.reset()
    st.rerun()

st.sidebar.markdown("---")

# 2. Document Focus Filter
st.sidebar.subheader("🎯 Chat Focus")
chat_target = st.sidebar.selectbox(
    "Choose what to chat with:",
    ["All Documents"] + existing_docs,
    index=0,
    help="Select a specific file to restrict the search context to that file only."
)

st.sidebar.markdown("---")

# 3. Document Manager (Knowledge Base)
st.sidebar.subheader("📁 Documents")
uploaded_files = st.sidebar.file_uploader("Upload .pdf, .md, or .txt", accept_multiple_files=True, label_visibility="collapsed")
if uploaded_files:
    for uploaded_file in uploaded_files:
        dest_path = os.path.join("docs", uploaded_file.name)
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s)")
    existing_docs = os.listdir("docs")

if st.sidebar.button("🔄 Index Uploaded Files", use_container_width=True):
    with st.sidebar.status("Indexing documents..."):
        try:
            chunks = run_ingestion(docs_dir="docs", db_path="./chroma_db", collection_name="my_notes")
            st.session_state.index_version += 1
            st.sidebar.success(f"Indexed {chunks} chunks!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# List uploaded files & delete buttons
if existing_docs:
    st.sidebar.markdown("<div style='margin-top: 1rem; font-size: 0.85rem; font-weight: 600; color: #4b5563;'>Files list:</div>", unsafe_allow_html=True)
    for doc in existing_docs:
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.markdown(f"<span style='font-size:0.8rem; color:#4b5563;'>📄 {doc}</span>", unsafe_allow_html=True)
        if col2.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
            try:
                os.remove(os.path.join("docs", doc))
                st.session_state.index_version += 1
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error deleting: {e}")
else:
    st.sidebar.info("No documents uploaded.")

st.sidebar.markdown("---")

# 4. Advanced Settings (Hidden in an Expander)
with st.sidebar.expander("⚙️ Advanced Settings"):
    available_models = get_installed_ollama_models()
    selected_model = st.selectbox("Ollama Model", available_models, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)
    similarity_top_k = st.slider("Retrieve Chunks (Top-K)", 1, 6, 3)
    sys_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, friendly AI assistant. Answer using the context documents when relevant. Otherwise, answer using your general knowledge.",
        height=100
    )

# --- Main App Layout ---
st.markdown("<div class='main-title'>Personal AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='main-subtitle'>Chat with your files in a clean, private interface</div>", unsafe_allow_html=True)

# Fetch current index
index, db_count = load_cached_index(st.session_state.index_version)

# Setup LLM globally on every rerun
Settings.llm = Ollama(model=selected_model, temperature=temperature, request_timeout=120.0)

# Setup LLM and Chat Engine dynamically if settings changed
if (
    "chat_engine" not in st.session_state
    or st.session_state.current_model != selected_model
    or st.session_state.current_top_k != similarity_top_k
    or st.session_state.current_temp != temperature
    or st.session_state.current_sys_prompt != sys_prompt
    or st.session_state.current_index_version != st.session_state.index_version
    or st.session_state.current_chat_target != chat_target
):
    st.session_state.current_model = selected_model
    st.session_state.current_top_k = similarity_top_k
    st.session_state.current_temp = temperature
    st.session_state.current_sys_prompt = sys_prompt
    st.session_state.current_index_version = st.session_state.index_version
    st.session_state.current_chat_target = chat_target
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    
    # Configure metadata filters if specific file selected
    filters = None
    if chat_target != "All Documents":
        filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=chat_target)])
    
    if index is None or db_count == 0:
        st.session_state.chat_engine = Settings.llm.as_chat_engine(
            system_prompt=sys_prompt,
            memory=memory
        )
    else:
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            similarity_top_k=similarity_top_k,
            memory=memory,
            system_prompt=sys_prompt,
            filters=filters
        )

# Render Chat History
if not st.session_state.messages:
    # Minimal welcome prompt
    st.markdown(f"""
    <div style="text-align:center; padding: 4rem 2rem; color:#6b7280;">
        <span style="font-size: 1.1rem;">👋 Hello! Ask me a question about your files or start a general chat.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 View Sources Used"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-meta">Source {idx+1}: {src['file_name']} (Relevance: {src['score']})</div>
                            <div class="source-text">{src['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)

# User Chat Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Stream assistant response
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        full_response = st.write_stream(response_stream.response_gen)
        
        # Extract sources
        sources = []
        if hasattr(response_stream, "source_nodes") and response_stream.source_nodes:
            seen_snippets = set()
            for node in response_stream.source_nodes:
                text_snippet = node.node.get_content().strip()
                if text_snippet not in seen_snippets:
                    seen_snippets.add(text_snippet)
                    score_val = f"{node.score:.3f}" if isinstance(node.score, float) else "N/A"
                    sources.append({
                        "file_name": node.metadata.get("file_name", "unknown"),
                        "score": score_val,
                        "text": text_snippet
                    })
                    
        # Render sources
        if sources:
            with st.expander("🔍 View Sources Used"):
                for idx, src in enumerate(sources):
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-meta">Source {idx+1}: {src['file_name']} (Relevance: {src['score']})</div>
                        <div class="source-text">{src['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
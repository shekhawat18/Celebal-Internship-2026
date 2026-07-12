"""
Professional Streamlit UI for the RAG Document Question Answering System

Run:
    streamlit run app.py
"""

import os
import streamlit as st
from src.rag_pipeline import RAGPipeline
import config

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System (RAG)")
st.markdown(
    "Ask questions about **your own PDFs or text files** using Retrieval-Augmented Generation."
)

# =====================================================
# SESSION STATE
# =====================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        os.makedirs(config.DATA_DIR, exist_ok=True)

        for file in uploaded_files:
            with open(
                os.path.join(config.DATA_DIR, file.name),
                "wb"
            ) as f:
                f.write(file.getbuffer())

        st.success(f"✅ {len(uploaded_files)} document(s) uploaded.")

    st.divider()

    st.header("🧠 Build Knowledge Base")

    if st.button(
        "🚀 Build / Rebuild Index",
        use_container_width=True
    ):

        with st.spinner("Processing documents..."):

            pipeline = RAGPipeline()

            pipeline.ingest()

            st.session_state.pipeline = pipeline

        st.success("Knowledge base created successfully!")

    st.divider()

    st.header("⚙ System Configuration")

    st.info(f"""
Embedding Model

{config.EMBEDDING_MODEL_NAME}

---

LLM Backend

{config.LLM_BACKEND}

---

Chunk Size

{config.CHUNK_SIZE}

---

Chunk Overlap

{config.CHUNK_OVERLAP}

---

Top K

{config.TOP_K}
""")

    # Metrics after ingestion

    if (
        st.session_state.pipeline is not None
        and st.session_state.pipeline.metrics
    ):

        st.divider()

        st.header("📊 System Metrics")

        metrics = st.session_state.pipeline.metrics

        st.metric("Documents", metrics["documents"])
        st.metric("Chunks", metrics["chunks"])
        st.metric("Characters", metrics["characters"])
        st.metric("Embedding Dim", metrics["embedding_dimension"])
        st.metric("Build Time", f"{metrics['ingestion_time']} sec")

# =====================================================
# MAIN
# =====================================================

st.header("💬 Ask Questions")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is the main idea of this document?"
)

if st.button(
    "🔍 Generate Answer",
    use_container_width=True
):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    if st.session_state.pipeline is None:

        with st.spinner("Loading vector database..."):

            pipeline = RAGPipeline()

            pipeline.load_index()

            st.session_state.pipeline = pipeline

    with st.spinner("Generating answer..."):

        result = st.session_state.pipeline.query(question)

    st.session_state.chat_history.append(result)

# =====================================================
# CHAT HISTORY
# =====================================================

for chat in reversed(st.session_state.chat_history):

    st.markdown("---")

    st.markdown("### 🙋 Question")

    st.write(chat["question"])

    st.markdown("### 🤖 Answer")

    st.success(chat["answer"])

    metrics = chat["metrics"]

    with st.expander("📊 Response Metrics"):

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Embedding",
                f"{metrics['embedding_time']} sec"
            )

            st.metric(
                "Retrieval",
                f"{metrics['retrieval_time']} sec"
            )

        with col2:

            st.metric(
                "Generation",
                f"{metrics['generation_time']} sec"
            )

            st.metric(
                "Total",
                f"{metrics['total_time']} sec"
            )

    with st.expander("📚 Retrieved Sources"):

        for source in chat["sources"]:

            st.markdown(
                f"### 📄 {source['source']}"
            )

            c1, c2 = st.columns(2)

            c1.metric("Chunk", source["chunk_id"])
            c2.metric("Similarity", f"{source['score']:.3f}")

            st.code(
                source["text"],
                language="text"
            )

            st.divider()

st.markdown("---")

st.caption(
    "Built using Sentence Transformers • FAISS • Streamlit • Retrieval-Augmented Generation"
)
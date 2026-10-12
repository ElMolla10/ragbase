"""Streamlit chat UI for RAGBase — modern redesign."""

import os

import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Custom CSS — dark, clean, modern
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Base */
    .stApp {
        background: #0f1117;
        color: #e6e6e6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161922;
        border-right: 1px solid #2a2d3a;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #a0a4b8;
    }

    /* Titles */
    h1, h2, h3 {
        font-family: "Inter", system-ui, -apple-system, sans-serif;
        font-weight: 650;
        letter-spacing: -0.02em;
    }
    h1 {
        background: linear-gradient(90deg, #7c9cff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.4rem;
        margin-bottom: 0.25rem;
    }

    /* Chat messages */
    .stChatMessage {
        background: #1a1d27 !important;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: #1e2433 !important;
        border-color: #3b4a6b;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 0.75rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
        color: #7c9cff;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 156, 255, 0.25);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1a1d27;
        border: 1px dashed #3b4a6b;
        border-radius: 12px;
        padding: 1.5rem;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: #1a1d27;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Captions & small text */
    .stCaption {
        color: #8b90a0 !important;
    }

    /* Dividers */
    hr {
        border-color: #2a2d3a;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #1a1d27;
        border-radius: 8px;
    }

    /* Success / info / error */
    .stSuccess, .stInfo, .stError, .stWarning {
        border-radius: 10px;
    }
</style>
"""


def copy_button(text: str) -> None:
    """Render a copy to clipboard button."""
    escaped_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{escaped_text}`).then(() => {{
            this.innerText = '✓ Copied';
            this.style.background = '#22c55e';
            setTimeout(() => {{
                this.innerText = 'Copy';
                this.style.background = '#1a1d27';
            }}, 1800);
        }})" style="
            background: #1a1d27;
            color: #c4c8d4;
            border: 1px solid #3b4a6b;
            border-radius: 6px;
            padding: 5px 14px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.15s;
        ">Copy</button>
        """,
        height=38,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_documents() -> list[dict]:
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch documents: {e}")
        return []


def upload_document(file) -> dict | None:
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            files=files,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to upload document: {e}")
        return None


def query_documents(query: str, document_id: int | None = None) -> dict | None:
    try:
        payload = {"query": query}
        if document_id is not None:
            payload["document_id"] = document_id

        response = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Query failed: {e}")
        return None


def delete_document(document_id: int) -> dict | None:
    try:
        response = requests.delete(
            f"{BACKEND_URL}/documents/{document_id}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to delete document: {e}")
        return None


def render_chat_page() -> None:
    st.markdown("### Chat with your papers")
    st.caption("Ask in English, Arabic, or mixed. Answers include source citations.")

    documents = get_documents()
    doc_options = {"All documents": None}
    doc_options.update({doc["filename"]: doc["id"] for doc in documents})

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        selected_doc = st.selectbox(
            "Scope",
            options=list(doc_options.keys()),
            label_visibility="collapsed",
        )
    selected_doc_id = doc_options[selected_doc]

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                meta = message["metadata"]
                copy_button(message["content"])
                st.caption(
                    f"{meta['latency_ms']} ms · {meta['token_count']} tokens · {meta['model_used']}"
                )
                if "sources" in meta and meta["sources"]:
                    with st.expander(f"Sources ({len(meta['sources'])})"):
                        for source in meta["sources"]:
                            st.markdown(
                                f"**Page {source['page_number']}**  ·  "
                                f"similarity `{source['similarity_score']:.3f}`"
                            )
                            st.markdown(f"> {source['content'][:280]}...")
                            st.divider()

    # Input
    if prompt := st.chat_input("Ask anything about the papers…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                result = query_documents(prompt, selected_doc_id)

            if result:
                st.markdown(result["answer"])
                copy_button(result["answer"])
                st.caption(
                    f"{result['latency_ms']} ms · {result['token_count']} tokens · {result['model_used']}"
                )

                if result.get("sources"):
                    with st.expander(f"Sources ({len(result['sources'])})"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**Page {source['page_number']}**  ·  "
                                f"similarity `{source['similarity_score']:.3f}`"
                            )
                            st.markdown(f"> {source['content'][:280]}...")
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "latency_ms": result["latency_ms"],
                            "token_count": result["token_count"],
                            "model_used": result["model_used"],
                            "sources": result["sources"],
                        },
                    }
                )


def render_documents_page() -> None:
    st.markdown("### Documents")
    st.caption("Upload research papers. They are chunked, embedded, and ready for questions.")

    st.markdown("#### Upload")
    uploaded_file = st.file_uploader(
        "Drop a PDF",
        type=["pdf"],
        help="Research paper or any PDF with selectable text",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("Process document", type="primary", use_container_width=True):
            with st.spinner("Extracting, chunking, embedding…"):
                result = upload_document(uploaded_file)
            if result:
                st.success(
                    f"**{result['filename']}** — {result['num_chunks']} chunks indexed"
                )
                st.rerun()

    st.markdown("#### Library")
    documents = get_documents()

    if not documents:
        st.info("No papers yet. Upload one above to get started.")
        return

    for doc in documents:
        with st.container():
            c1, c2, c3, c4 = st.columns([4, 1.2, 1.2, 1])
            with c1:
                st.markdown(f"**{doc['filename']}**")
            with c2:
                st.caption(f"{doc['num_chunks']} chunks")
            with c3:
                status = doc["status"]
                color = "#22c55e" if status == "completed" else "#f59e0b"
                st.markdown(
                    f"<span style='color:{color}; font-weight:500;'>{status}</span>",
                    unsafe_allow_html=True,
                )
            with c4:
                if st.button("Delete", key=f"del_{doc['id']}", type="secondary"):
                    if delete_document(doc["id"]):
                        st.success(f"Deleted {doc['filename']}")
                        st.rerun()
            st.divider()


def main() -> None:
    st.set_page_config(
        page_title="RAGBase",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        """
        <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:0.5rem;">
            <h1 style="margin:0;">RAGBase</h1>
            <span style="color:#8b90a0; font-size:1rem;">Research paper Q&A</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            options=["Chat", "Documents"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### Stats")
        try:
            response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
            if response.ok:
                stats = response.json()
                st.metric("Queries", stats["total_queries"])
                st.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
                st.metric("Documents", stats["total_documents"])
                st.metric("Chunks", stats["total_chunks"])
            else:
                st.caption("Stats unavailable")
        except requests.RequestException:
            st.caption("Could not load stats")

        st.divider()
        st.caption("Powered by FastAPI · pgvector · Groq")

    if page == "Chat":
        render_chat_page()
    else:
        render_documents_page()


if __name__ == "__main__":
    main()

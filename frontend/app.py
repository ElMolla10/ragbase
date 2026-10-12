"""Streamlit UI for RAGBase — premium redesign."""

import os

import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #09090b;
    --surface: #121214;
    --surface-2: #1c1c1f;
    --border: #27272a;
    --border-strong: #3f3f46;
    --text: #fafafa;
    --text-muted: #a1a1aa;
    --accent: #6366f1;
    --accent-soft: rgba(99, 102, 241, 0.15);
    --success: #22c55e;
    --warning: #f59e0b;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.main .block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 980px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* Hide default Streamlit branding noise */
#MainMenu, footer, header {visibility: hidden;}

/* Titles */
h1 {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.15rem 0 !important;
}
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: var(--text) !important;
}

/* Chat messages */
.stChatMessage {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
.stChatMessage[data-testid="stChatMessage-user"] {
    background: linear-gradient(180deg, #1a1a2e 0%, var(--surface) 100%) !important;
    border-color: #3730a3;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 0.9rem;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 650 !important;
    color: #a5b4fc !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 550;
    border: 1px solid var(--border-strong);
    background: var(--surface-2);
    color: var(--text);
    transition: all 0.12s ease;
}
.stButton > button:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
}
.stButton > button[kind="primary"]:hover {
    background: #4f46e5;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1.5px dashed var(--border-strong);
    border-radius: 14px;
    padding: 1.75rem 1.25rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* Select */
.stSelectbox > div > div {
    background: var(--surface-2);
    border-radius: 10px;
    border-color: var(--border);
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface-2);
    border-radius: 10px;
    font-weight: 550;
    border: 1px solid var(--border);
}

/* Captions */
.stCaption, [data-testid="stCaption"] {
    color: var(--text-muted) !important;
}

/* Dividers */
hr { border-color: var(--border); opacity: 0.6; }

/* Success / info */
.stSuccess, .stInfo {
    border-radius: 12px;
    border: none;
}

/* Custom card class via markdown */
.doc-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
</style>
"""


def copy_button(text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{escaped}`).then(() => {{
            const orig = this.innerText;
            this.innerText = '✓ Copied';
            this.style.background = '#166534';
            this.style.borderColor = '#22c55e';
            setTimeout(() => {{
                this.innerText = orig;
                this.style.background = '#1c1c1f';
                this.style.borderColor = '#3f3f46';
            }}, 1600);
        }})" style="
            background:#1c1c1f; color:#e4e4e7; border:1px solid #3f3f46;
            border-radius:8px; padding:6px 14px; cursor:pointer;
            font-size:13px; font-weight:500; font-family:Inter,system-ui,sans-serif;
        ">Copy</button>
        """,
        height=36,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_documents() -> list[dict]:
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Could not load documents: {e}")
        return []


def upload_document(file) -> dict | None:
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        r = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None


def query_documents(query: str, document_id: int | None = None) -> dict | None:
    try:
        payload = {"query": query}
        if document_id is not None:
            payload["document_id"] = document_id
        r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Query failed: {e}")
        return None


def delete_document(document_id: int) -> dict | None:
    try:
        r = requests.delete(f"{BACKEND_URL}/documents/{document_id}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Delete failed: {e}")
        return None


def render_chat_page() -> None:
    st.markdown("#### Ask your papers")
    st.caption("English, Arabic, or mixed. Every answer cites the source pages.")

    documents = get_documents()
    doc_options = {"All papers": None}
    doc_options.update({d["filename"]: d["id"] for d in documents})

    c1, _ = st.columns([1.6, 3])
    with c1:
        selected = st.selectbox(
            "Scope",
            list(doc_options.keys()),
            label_visibility="collapsed",
        )
    doc_id = doc_options[selected]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg:
                m = msg["metadata"]
                copy_button(msg["content"])
                st.caption(
                    f"{m['latency_ms']} ms  ·  {m['token_count']} tokens  ·  {m['model_used']}"
                )
                sources = m.get("sources") or []
                if sources:
                    with st.expander(f"Sources · {len(sources)}"):
                        for s in sources:
                            st.markdown(
                                f"**Page {s['page_number']}**  ·  "
                                f"`{s['similarity_score']:.3f}`"
                            )
                            st.markdown(f"> {s['content'][:260]}…")
                            st.divider()

    if prompt := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = query_documents(prompt, doc_id)
            if result:
                st.markdown(result["answer"])
                copy_button(result["answer"])
                st.caption(
                    f"{result['latency_ms']} ms  ·  {result['token_count']} tokens  ·  {result['model_used']}"
                )
                sources = result.get("sources") or []
                if sources:
                    with st.expander(f"Sources · {len(sources)}"):
                        for s in sources:
                            st.markdown(
                                f"**Page {s['page_number']}**  ·  "
                                f"`{s['similarity_score']:.3f}`"
                            )
                            st.markdown(f"> {s['content'][:260]}…")
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "latency_ms": result["latency_ms"],
                            "token_count": result["token_count"],
                            "model_used": result["model_used"],
                            "sources": sources,
                        },
                    }
                )


def render_documents_page() -> None:
    st.markdown("#### Your library")
    st.caption("Upload a PDF. It’s chunked, embedded, and ready to query.")

    st.markdown("##### Upload")
    uploaded = st.file_uploader(
        "PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Any PDF with selectable text",
    )
    if uploaded and st.button("Index document", type="primary", use_container_width=True):
        with st.spinner("Extracting · chunking · embedding…"):
            res = upload_document(uploaded)
        if res:
            st.success(f"**{res['filename']}** indexed · {res['num_chunks']} chunks")
            st.rerun()

    st.markdown("##### Papers")
    docs = get_documents()
    if not docs:
        st.info("No papers yet. Upload one above.")
        return

    for doc in docs:
        status = doc["status"]
        color = "#22c55e" if status == "completed" else "#f59e0b"
        c1, c2, c3 = st.columns([5, 1.4, 1])
        with c1:
            st.markdown(f"**{doc['filename']}**")
            st.caption(f"{doc['num_chunks']} chunks")
        with c2:
            st.markdown(
                f"<span style='color:{color};font-weight:600;font-size:0.9rem;'>{status}</span>",
                unsafe_allow_html=True,
            )
        with c3:
            if st.button("Delete", key=f"d{doc['id']}", type="secondary"):
                if delete_document(doc["id"]):
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
        <div style="margin-bottom:1.25rem;">
            <h1>RAGBase</h1>
            <p style="color:#a1a1aa;margin:0;font-size:1.05rem;">
                Research paper Q&A with citations and observability
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    with st.sidebar:
        st.markdown("##### Navigation")
        page = st.radio(
            "nav",
            ["Chat", "Documents"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("##### Stats")
        try:
            r = requests.get(f"{BACKEND_URL}/stats", timeout=5)
            if r.ok:
                s = r.json()
                st.metric("Queries", s["total_queries"])
                st.metric("Avg latency", f"{s['avg_latency_ms']:.0f} ms")
                st.metric("Documents", s["total_documents"])
                st.metric("Chunks", s["total_chunks"])
            else:
                st.caption("Stats unavailable")
        except requests.RequestException:
            st.caption("Could not load stats")

        st.divider()
        st.caption("FastAPI · pgvector · Groq")

    if page == "Chat":
        render_chat_page()
    else:
        render_documents_page()


if __name__ == "__main__":
    main()

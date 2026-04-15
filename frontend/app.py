"""Streamlit chat UI for RAGBase."""

import os

import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def copy_button(text: str, key: str) -> None:
    """Render a copy to clipboard button."""
    # Escape text for JavaScript
    escaped_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{escaped_text}`).then(() => {{
            this.innerText = 'Copied!';
            setTimeout(() => this.innerText = 'Copy to Clipboard', 2000);
        }})" style="
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #4a4a5a;
            border-radius: 4px;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 14px;
        ">Copy to Clipboard</button>
        """,
        height=40,
        key=key,
    )


def init_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_documents() -> list[dict]:
    """Fetch list of documents from backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch documents: {e}")
        return []


def upload_document(file) -> dict | None:
    """Upload a document to the backend."""
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
    """Send a query to the backend."""
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


def render_chat_page() -> None:
    """Render the chat interface."""
    st.header("Chat with Research Papers")

    # Document selector
    documents = get_documents()
    doc_options = {"All Documents": None}
    doc_options.update({doc["filename"]: doc["id"] for doc in documents})

    selected_doc = st.selectbox(
        "Select document to query",
        options=list(doc_options.keys()),
    )
    selected_doc_id = doc_options[selected_doc]

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                meta = message["metadata"]
                copy_button(message["content"], key=f"copy_history_{idx}")
                st.caption(
                    f"Latency: {meta['latency_ms']}ms | "
                    f"Tokens: {meta['token_count']} | "
                    f"Model: {meta['model_used']}"
                )
                if "sources" in meta:
                    with st.expander("View Sources"):
                        for source in meta["sources"]:
                            st.markdown(
                                f"**Page {source['page_number']}** "
                                f"(similarity: {source['similarity_score']:.4f})"
                            )
                            st.text(source["content"][:300] + "...")
                            st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about the research papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_documents(prompt, selected_doc_id)

            if result:
                st.markdown(result["answer"])
                copy_button(result["answer"], key="copy_live_response")
                st.caption(
                    f"Latency: {result['latency_ms']}ms | "
                    f"Tokens: {result['token_count']} | "
                    f"Model: {result['model_used']}"
                )

                with st.expander("View Sources"):
                    for source in result["sources"]:
                        st.markdown(
                            f"**Page {source['page_number']}** "
                            f"(similarity: {source['similarity_score']:.4f})"
                        )
                        st.text(source["content"][:300] + "...")
                        st.divider()

                # Add to history
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
    """Render the documents management page."""
    st.header("Document Management")

    # Upload section
    st.subheader("Upload New Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a research paper in PDF format",
    )

    if uploaded_file is not None and st.button("Process Document"):
        with st.spinner("Processing document..."):
            result = upload_document(uploaded_file)
        if result:
            st.success(
                f"Uploaded '{result['filename']}' with {result['num_chunks']} chunks"
            )
            st.rerun()

    # Documents list
    st.subheader("Uploaded Documents")
    documents = get_documents()

    if not documents:
        st.info("No documents uploaded yet. Upload a PDF to get started.")
    else:
        for doc in documents:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(doc["filename"])
                with col2:
                    st.text(f"{doc['num_chunks']} chunks")
                with col3:
                    status_color = "green" if doc["status"] == "completed" else "orange"
                    st.markdown(f":{status_color}[{doc['status']}]")
                st.divider()


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="RAGBase",
        page_icon="📚",
        layout="wide",
    )

    st.title("RAGBase - Research Paper Q&A")

    init_session_state()

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        options=["Chat", "Documents"],
        index=0,
    )

    # Stats in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Statistics")
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if response.ok:
            stats = response.json()
            st.sidebar.metric("Total Queries", stats["total_queries"])
            st.sidebar.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
            st.sidebar.metric("Documents", stats["total_documents"])
            st.sidebar.metric("Chunks", stats["total_chunks"])
    except requests.RequestException:
        st.sidebar.warning("Could not load stats")

    # Render selected page
    if page == "Chat":
        render_chat_page()
    else:
        render_documents_page()


if __name__ == "__main__":
    main()

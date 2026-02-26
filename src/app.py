import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv
from ingest import load_documents, split_documents, build_vectorstore
from chain import ask

load_dotenv()

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Document Assistant")
st.caption("Upload a document and ask anything about it — powered by Groq + LLaMA 3")

# ── Session State ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

# ── Sidebar: File Upload ──────────────────────────────
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file and not st.session_state.doc_loaded:
        # Save uploaded file to data/
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            docs   = load_documents("data/")
            chunks = split_documents(docs)
            build_vectorstore(chunks)
            st.session_state.doc_loaded = True

        st.success(f"✅ '{uploaded_file.name}' is ready!")

    if st.session_state.doc_loaded:
        st.info("Document loaded. Ask your questions!")

    st.divider()
    if st.button("🗑️ Reset"):
        st.session_state.messages  = []
        st.session_state.doc_loaded = False
        st.rerun()

# ── Chat Interface ────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your document..."):
    if not st.session_state.doc_loaded:
        st.warning("Please upload a document first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
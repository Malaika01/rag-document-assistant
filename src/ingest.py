import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Config ──────────────────────────────────────────
DATA_DIR        = "data/"
VECTORSTORE_DIR = "vectorstore/"
EMBED_MODEL     = "all-MiniLM-L6-v2"  # fast, lightweight, free

# ── Load Documents ───────────────────────────────────
def load_documents(data_dir: str):
    docs = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue
        docs.extend(loader.load())
        print(f"Loaded: {filename}")
    return docs

# ── Split into Chunks ────────────────────────────────
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# ── Embed & Store in FAISS ───────────────────────────
def build_vectorstore(chunks):
    print("Loading embedding model (first run may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vectorstore saved to {VECTORSTORE_DIR}")

# ── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting ingestion...")
    docs   = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("Done! Your documents are ready to query.")
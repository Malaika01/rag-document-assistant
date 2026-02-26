from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore/"
EMBED_MODEL     = "all-MiniLM-L6-v2"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_chunks(query: str, k: int = 4):
    """Take a question, return the top-k most relevant chunks."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results

# ── Quick Test ───────────────────────────────────────
if __name__ == "__main__":
    query = "What is this document about?"  # change this to test
    chunks = retrieve_chunks(query)
    
    print(f"\nTop {len(chunks)} relevant chunks for: '{query}'\n")
    print("─" * 50)
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(chunk.page_content)
        print("─" * 50)
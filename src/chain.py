import os
from dotenv import load_dotenv
from groq import Groq
from retriever import retrieve_chunks
import streamlit as st


load_dotenv()

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

client = Groq(api_key=get_api_key())
MODEL = "llama-3.1-8b-instant"  # fast & free, can upgrade to llama-3.1-70b

# ── Build Prompt ─────────────────────────────────────
def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    return prompt

# ── Ask the LLM ──────────────────────────────────────
def ask(question: str) -> str:
    chunks  = retrieve_chunks(question, k=4)
    prompt  = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful document assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

# ── Quick Test ───────────────────────────────────────
if __name__ == "__main__":
    question = "What is this document about?"  # change this to test
    print(f"Question: {question}\n")
    answer = ask(question)
    print(f"Answer:\n{answer}")
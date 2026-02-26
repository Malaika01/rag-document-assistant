# 📄 RAG Document Assistant

An AI-powered document assistant that lets you upload any PDF or text file and ask questions about it in natural language — powered by **Groq (LLaMA 3.1)**, **LangChain**, and **FAISS**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🧠 How It Works
```
User uploads document
        ↓
Document split into chunks → embedded locally (sentence-transformers)
        ↓
Chunks stored in FAISS vector database
        ↓
User asks a question → question embedded → similarity search
        ↓
Top relevant chunks retrieved → stuffed into LLM prompt
        ↓
Groq (LLaMA 3.1) generates a grounded answer
```

---

## ✨ Features

- 📄 Upload PDF or TXT documents via a clean chat UI
- 🔍 Semantic search using FAISS vector store
- 🤖 Answers grounded in your document (no hallucination)
- ⚡ Blazing fast responses via Groq's inference API
- 🔒 Fully local embeddings — your documents never leave your machine

---

## 🛠️ Tech Stack

| Layer         | Tool                          |
|---------------|-------------------------------|
| LLM           | Groq API (LLaMA 3.1 8B/70B)  |
| Embeddings    | sentence-transformers (local) |
| Vector Store  | FAISS                         |
| Orchestration | LangChain                     |
| UI            | Streamlit                     |
| Doc Parsing   | PyMuPDF                       |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-document-assistant.git
cd rag-document-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
cd src
streamlit run app.py
```

---

## 📁 Project Structure
```
rag-document-assistant/
├── data/               # uploaded documents (gitignored)
├── vectorstore/        # FAISS index (gitignored)
├── src/
│   ├── ingest.py       # load, chunk, embed, store
│   ├── retriever.py    # semantic search
│   ├── chain.py        # RAG chain + Groq LLM
│   └── app.py          # Streamlit UI
├── .env                # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## 📸 Demo
https://rag-document-assistant-gobu9uhuurk6jph7hyfcjo.streamlit.app/ 

---
# HR Policy Assistant (RAG-based)

This project is a Retrieval-Augmented Generation (RAG) based HR assistant built using:

- LangChain
- FAISS (vector database)
- Ollama (local LLM)
- HuggingFace Embeddings

## 📌 What it does

You can ask HR-related questions like:

- "What is the leave policy?"
- "What is the notice period?"
- "Can I work from home?"

The system retrieves relevant data from HR documents and generates accurate answers.

---

## ⚙️ Tech Stack

- Python
- LangChain
- FAISS (local vector store)
- Ollama (Llama3 model)
- HuggingFace embeddings

---

## 🧠 How it works

1. Loads HR policy document
2. Splits into chunks
3. Converts text → embeddings
4. Stores in FAISS vector DB
5. Retrieves relevant chunks
6. Sends to LLM (Ollama)
7. Generates answer

---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers

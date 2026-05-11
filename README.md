# MyRAGSystem 🚀

An autonomous Retrieval-Augmented Generation (RAG) system built with **NEO**. This project features semantic search, multilingual support, and interactive vector space visualization.

## ✨ Features
- **Local RAG Pipeline:** Processes PDF and Markdown files locally.
- **Multilingual Support:** Uses `paraphrase-multilingual-MiniLM-L12-v2` for high-accuracy search in Ukrainian and other languages.
- **Semantic Search:** Powered by **FAISS** (Facebook AI Similarity Search) for near-instant retrieval.
- **LLM Integration:** Synthesizes natural language answers based on retrieved context.
- **Interactive UI:** A multi-tab **Streamlit** dashboard for searching and visualizing the knowledge base.
- **Vector Visualization:** 2D mapping of document embeddings to see how your data is clustered.

## 🛠️ Tech Stack
- **Python**
- **Sentence-Transformers** (Embeddings)
- **FAISS** (Vector Store)
- **Streamlit** (Frontend)
- **PyPDF2 / Markdown** (Data Extraction)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Optional] Ollama for local LLM inference

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/SergiySvirkov/MyRAGSystem.git](https://github.com/SergiySvirkov/MyRAGSystem.git)
   Install dependencies:
   pip install -r requirements.txt

   Usage

Run the application:

streamlit run app.py

Project Structure

    app.py: Main Streamlit interface.

    vector_store.py: FAISS index management.

    embeddings.py: Multilingual embedding logic.

    search.py: Similarity search and LLM synthesis.
   cd MyRAGSystem

   

# Agentic AI eBook — RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions **strictly**
from the *"Agentic AI for Executives"* eBook,
using:

- **FAISS** for vector storage & similarity search
- **sentence-transformers** (`all-MiniLM-L6-v2`) for local, free text embeddings
- **LangGraph** to orchestrate the `retrieve → generate` pipeline as a graph
- **Llama 3.3 70B** served by the **Groq API** for fast generation
- **Streamlit** for the chat UI

## 🌐 Live App

👉 [Open the Agentic AI eBook — RAG Chatbot](https://agenticaiebook-rag-chatbot-cxb5a4f5bgyxok5unf4q8g.streamlit.app/)

## 📁 Project Structure

```
agentic-ai-rag-chatbot/
├── app.py                 # Streamlit chat UI
├── config.py               # Central configuration (paths, model names, params)
├── ingest.py                # PDF → chunks → embeddings → FAISS index (whenever source PDF changes)
├── raggraph.py               # LangGraph RAG pipeline (retrieve + generate nodes)
├── requirements.txt
├── README.md
├── .env                      # Your GROQ_API_KEY goes here
├── data/
│   └── Ebook-Agentic-AI.pdf   # Source knowledge base
└── vector_store/              # Created by ingest.py (FAISS index + metadata)
```

## 🚀 Setup

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Add your Groq API key

Get a free key from https://console.groq.com, then edit `.env`:

```
GROQ_API_KEY=gsk_your_real_key_here
```

### 3. Make sure the eBook PDF is in place

The PDF should already be at `data/Ebook-Agentic-AI.pdf`. If it's missing,
copy your copy of the eBook there (keep the same filename, or update
`PDF_PATH` in `config.py`).

### 4. Build the vector index (one-time step)

```bash
python ingest.py
```

This will:
1. Load the PDF page by page
2. Split it into ~800-character overlapping chunks
3. Embed every chunk with `all-MiniLM-L6-v2`
4. Save a FAISS index + chunk metadata to `vector_store/`

Re-run this whenever source PDF changes

### 5. Launch the chatbot

```bash
streamlit run app.py
```

## 💬 How it works (RAG pipeline)

```
User question
     │
     ▼
┌─────────────┐   embed question, similarity_search_with_score(k=4)
│  retrieve   │──────────────────────────────────────────────────►  FAISS index
└─────────────┘
     │  top-K chunks + similarity scores
     ▼
┌─────────────┐   if avg similarity < threshold → refuse to answer
│  generate   │   else → build strict "context-only" prompt
└─────────────┘   → call Llama 3.3 70B via Groq
     │
     ▼
Final answer + retrieved chunks + confidence score
```

## 💬 Sample Queries

Try asking the chatbot:

1. What is Agentic AI?
2. What are roles of an agents?
3. What are the key characteristics of an AI agent?
4. How does Agentic AI improve business process automation?
5. What should organizations consider before adopting Agentic AI?
6. Who won the 2026 FIFA World Cup?

## 🖥️ What the UI shows for every answer

- **Final answer** — the grounded response from Llama 3.3 70B
- **Retrieved context chunks** — the exact eBook passages used, with page
  numbers and per-chunk similarity scores (in a collapsible expander)
- **Confidence score** — the average similarity score across retrieved
  chunks, shown as a progress bar (0 = no relevant match, 1 = very close
  match)


## 🔁 Re-ingesting after changes

Any time you edit `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL_NAME`, or
swap the PDF, delete `vector_store/` and re-run `python ingest.py`.

## 🧩 Notes

- The FAISS index is loaded with `allow_dangerous_deserialization=True`
  because it's a local pickle you generated yourself via `ingest.py` — don't
  load FAISS indexes you didn't create.
- `st.cache_resource` in `app.py` ensures the embedding model, FAISS index,
  and Groq client are loaded only once per Streamlit session, not on every
  question.

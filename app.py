import streamlit as st

import config
from raggraph import RAGPipeline

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Agentic AI eBook — Chatbot")
st.caption(
    "Ask questions about the *Agentic AI for Executives* eBook."
)


@st.cache_resource(show_spinner="Loading vector index and LLM ...")
def load_pipeline():
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python ingest.py` first to build the FAISS index.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.info("Add your GROQ_API_KEY to the .env file (see .env.example).")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


def render_turn(question: str, result: dict):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(result["answer"])

        conf = float(result["confidence"])
        st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence score: {conf:.2f}")

        with st.expander(f"📄 Retrieved context chunks ({len(result['chunks'])})"):
            if not result["chunks"]:
                st.write("No chunks retrieved.")
            for i, chunk in enumerate(result["chunks"]):
                st.markdown(
                    f"**Chunk {i + 1}** — page {chunk['page']} — "
                    f"similarity score: `{chunk['score']}`"
                )
                st.code(chunk["content"], language=None)


# Replay history
for turn in st.session_state.history:
    render_turn(turn["question"], turn["result"])

# New input
question = st.chat_input("Ask something about Agentic AI ...")

if question:
    with st.spinner("Retrieving context and generating answer ..."):
        result = pipeline.run(question)

    render_turn(question, result)
    st.session_state.history.append({"question": question, "result": result})

with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**LLM:** `{config.GROQ_MODEL_NAME}` (Groq)")
    st.write(f"**Embedding model:** `{config.EMBEDDING_MODEL_NAME}`")
    st.write(f"**Top-K retrieved chunks:** {config.TOP_K}")
    st.write(f"**Confidence threshold:** {config.SIMILARITY_SCORE_THRESHOLD}")
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.rerun()

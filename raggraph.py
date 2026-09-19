import os
import config
from typing import List, TypedDict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores.utils import DistanceStrategy


SYSTEM_PROMPT = """You are a helpful assistant that answers questions STRICTLY using \
the provided context, which is extracted from the "Agentic AI for Executives" eBook.

Rules you must follow:
1. Only use information found in the context below. Never use outside knowledge.
2. If the answer is not contained in the context, respond EXACTLY with:
   "I don't have enough information in the eBook to answer that question."
3. Be concise, accurate, and faithful to the eBook's wording and intent.
4. Never invent page numbers, statistics, names, or facts that are not present \
in the context.

Context: {context}
"""

USER_PROMPT = "Question: {question}"


class RAGState(TypedDict):
    question: str
    retrieved_docs: list
    scores: list
    context: str
    answer: str
    confidence: float


class RAGPipeline:

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to your .env file "
            )

        print("[raggraph] Loading embedding model ...")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

        if not os.path.isdir(config.INDEX_PATH):
            raise FileNotFoundError(
                f"No FAISS index found at '{config.INDEX_PATH}'. "
                f"Run `python ingest.py` first to build it."
            )

        print("[raggraph] Loading FAISS index ...")
        self.vectorstore = FAISS.load_local(
            config.INDEX_PATH,
            self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True,
        )

        print(f"[raggraph] Connecting to Groq model: {config.GROQ_MODEL_NAME} ...")
        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )

        self.graph = self._build_graph()


    # Graph nodes

    def _retrieve_node(self, state: RAGState) -> RAGState:
        question = state["question"]
        results = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=config.TOP_K
        )
        docs = [r[0] for r in results]
        raw_distances = [r[1] for r in results]
        similarities = [1.0 / (1.0 + d) for d in raw_distances]

        context = "\n\n---\n\n".join(
            f"[Chunk {i + 1} | page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for i, d in enumerate(docs)
        )

        return {**state, "retrieved_docs": docs, "scores": similarities, "context": context}

    def _generate_node(self, state: RAGState) -> RAGState:
        similarities = state["scores"]
        avg_confidence = sum(similarities) / len(similarities) if similarities else 0.0

        if not state["retrieved_docs"] or avg_confidence < config.SIMILARITY_SCORE_THRESHOLD:
            return {
                **state,
                "answer": "I don't have enough information in the eBook to answer that question.",
                "confidence": 0.0,
            }

        messages = self.prompt.format_messages(
            context=state["context"], question=state["question"]
        )
        response = self.llm.invoke(messages)
        answer = response.content

        final_confidence = 0.0 if answer.strip() == "I don't have enough information in the eBook to answer that question." else round(avg_confidence, 3)

        return {**state, "answer": response.content, "confidence": final_confidence}

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()


    # Public API

    def run(self, question: str) -> dict:
        """Runs the full retrieve -> generate graph for a single question.

        Returns a dict with:
            answer:     the grounded final answer
            chunks:     list of {content, page, score} dicts, in retrieval order
            confidence: average similarity score across retrieved chunks (0-1)
        """
        initial_state: RAGState = {
            "question": question,
            "retrieved_docs": [],
            "scores": [],
            "context": "",
            "answer": "",
            "confidence": 0.0,
        }
        final_state = self.graph.invoke(initial_state)

        chunks_out = [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "score": round(score, 3),
            }
            for doc, score in zip(final_state["retrieved_docs"], final_state["scores"])
        ]

        return {
            "answer": final_state["answer"],
            "chunks": chunks_out,
            "confidence": final_state["confidence"],
        }

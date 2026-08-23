"""
Two Chroma collections, deliberately kept separate:

1. curriculum      -> verified textbook/course content. This is what answers
                       must stay grounded in (faithfulness is checked against
                       chunks retrieved from HERE).
2. misconceptions   -> a small hand/curated library of known student error
                       patterns per topic (e.g. "flips inequality sign
                       incorrectly when dividing by negative"). This is what
                       makes the tutor's diagnosis specific instead of a
                       generic "try again."

Most student RAG-tutor projects only build #1. #2 is the differentiator.
"""
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from app.config import settings

_embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)


def get_curriculum_store() -> Chroma:
    return Chroma(
        collection_name=settings.CURRICULUM_COLLECTION,
        embedding_function=_embeddings,
        persist_directory=settings.CHROMA_DIR,
    )


def get_misconception_store() -> Chroma:
    return Chroma(
        collection_name=settings.MISCONCEPTION_COLLECTION,
        embedding_function=_embeddings,
        persist_directory=settings.CHROMA_DIR,
    )


def get_problem_bank_store() -> Chroma:
    """
    Stores known problems paired with a canonical answer that was computed
    independently (via sympy, not the LLM) — see app/problem_bank.py.
    This is what lets app/answer_leak.py verify a draft reply against a
    ground-truth answer instead of just trusting the LLM followed the
    'don't reveal the answer' instruction.
    """
    return Chroma(
        collection_name=settings.PROBLEM_BANK_COLLECTION,
        embedding_function=_embeddings,
        persist_directory=settings.CHROMA_DIR,
    )


def _load_file(path: str):
    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()
    if path.endswith(".docx"):
        return Docx2txtLoader(path).load()
    return TextLoader(path, encoding="utf-8").load()


def ingest_curriculum_file(path: str, source_label: str) -> int:
    """Load, chunk, and embed one verified curriculum source file."""
    docs = _load_file(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = source_label
    store = get_curriculum_store()
    store.add_documents(chunks)
    return len(chunks)


def seed_misconceptions(topic: str, misconceptions: List[dict]):
    """
    misconceptions: [{"pattern": "...", "explanation": "...", "targeted_hint": "..."}]
    Call this once per topic to build up the misconception library.
    Example entries live in scripts/seed_misconceptions.py (not shown here).
    """
    store = get_misconception_store()
    texts = [m["pattern"] for m in misconceptions]
    metadatas = [
        {
            "topic": topic,
            "explanation": m["explanation"],
            "targeted_hint": m["targeted_hint"],
        }
        for m in misconceptions
    ]
    store.add_texts(texts=texts, metadatas=metadatas)

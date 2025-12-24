"""
ingest.py

Reads data/faqs.txt, creates embeddings, builds a FAISS index and saves it to disk.

Usage:
    python ingest.py         # uses real embeddings if available, otherwise falls back to mock
    python ingest.py --mock  # force mock mode (no API keys required)
"""

import argparse
import os
import pickle
from typing import List

try:
    from langchain.docstore.document import Document
    from langchain.vectorstores import FAISS
    from langchain.embeddings.base import Embeddings
    from langchain.embeddings import HuggingFaceEmbeddings
except Exception:
    # We'll still provide helpful error messages later if imports fail
    Document = None
    FAISS = None
    Embeddings = object  # type: ignore

import hashlib
import random
import numpy as np


FAQ_PATH = os.path.join("data", "faqs.txt")
INDEX_DIR = "faiss_index"
DOCS_PICKLE = os.path.join(INDEX_DIR, "docs.pkl")


class MockEmbeddings(Embeddings):
    """Deterministic mock embeddings (works offline).

    Converts text into a fixed-size vector using a seeded pseudo-random generator
    based on a hash of the text. This is NOT semantically meaningful but useful
    for local testing and demo without API keys.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _text_to_vector(self, text: str) -> List[float]:
        # Use md5 to get a stable integer seed from the text
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16)
        rnd = random.Random(seed)
        vec = [rnd.random() for _ in range(self.dim)]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)


def load_faqs(path: str) -> List[str]:
    """Load FAQs from a text file. Each FAQ item should be separated by a blank line."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Split on two or more newlines to separate entries
    entries = [e.strip() for e in raw.split("\n\n") if e.strip()]
    return entries


def create_documents(faq_texts: List[str]):
    """Create LangChain Document objects from the FAQ strings."""
    docs = []
    for i, text in enumerate(faq_texts):
        metadata = {"source": f"faq_{i+1}"}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def get_embeddings(mock: bool):
    if mock:
        print("Using mock embeddings (offline).")
        return MockEmbeddings()
    # Try to use HuggingFace sentence-transformers if available
    try:
        print("Attempting to use HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2).")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print("Could not initialize HuggingFace embeddings:", str(e))
        print("Falling back to mock embeddings.")
        return MockEmbeddings()


def build_and_save_index(docs, embeddings, index_dir: str = INDEX_DIR):
    """Build a FAISS index from documents and save it to disk."""
    if FAISS is None:
        raise RuntimeError("Required packages are missing. Please install requirements.txt first.")

    print("Creating FAISS index from documents...")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    os.makedirs(index_dir, exist_ok=True)
    print(f"Saving FAISS index to '{index_dir}' ...")
    vectorstore.save_local(folder_path=index_dir)
    # Save the original docs so the chatbot can show sources or metadata
    with open(DOCS_PICKLE, "wb") as f:
        pickle.dump(docs, f)
    print("Index and documents saved.")


def main():
    parser = argparse.ArgumentParser(description="Ingest FAQs and build FAISS index.")
    parser.add_argument("--mock", action="store_true", help="Use mock embeddings (no APIs).")
    parser.add_argument("--faq-path", default=FAQ_PATH, help="Path to faqs.txt")
    parser.add_argument("--index-dir", default=INDEX_DIR, help="Directory to save the FAISS index")
    args = parser.parse_args()

    if not os.path.exists(args.faq_path):
        print(f"FAQ file not found: {args.faq_path}")
        return

    faqs = load_faqs(args.faq_path)
    print(f"Loaded {len(faqs)} FAQ entries.")

    docs = create_documents(faqs)
    embeddings = get_embeddings(mock=args.mock)

    build_and_save_index(docs, embeddings, index_dir=args.index_dir)
    print("Done.")


if __name__ == "__main__":
    main()

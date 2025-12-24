"""
chatbot.py

Simple command-line RAG FAQ chatbot.

Usage:
    python chatbot.py         # will try to use OpenAI (if OPENAI_API_KEY is set) or HuggingFaceHub (if HF token set)
    python chatbot.py --mock  # force mock mode (works offline)
"""

import argparse
import os
import pickle
from typing import List

try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings.base import Embeddings
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import OpenAI, HuggingFaceHub
except Exception:
    FAISS = None
    Embeddings = object  # type: ignore

from ingest import MockEmbeddings, get_embeddings, DOCS_PICKLE, INDEX_DIR
import textwrap


def get_llm(mock: bool):
    """Return a callable LLM. If mock is True or no keys found, return a MockLLM."""
    if mock:
        print("Running in MOCK LLM mode. Responses are generated without API calls.")
        return MockLLM()

    # Prefer OpenAI if API key is present
    if os.getenv("OPENAI_API_KEY"):
        try:
            print("Using OpenAI LLM.")
            return OpenAI(temperature=0)
        except Exception as e:
            print("Failed to initialize OpenAI LLM:", e)

    # Try HuggingFace Hub if token present
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        try:
            print("Using HuggingFaceHub LLM.")
            # Model can be changed; choose a small sequence model that works with the Hub
            return HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length": 256})
        except Exception as e:
            print("Failed to initialize HuggingFaceHub LLM:", e)

    # Fall back to mock
    print("No LLM API key found. Falling back to MOCK LLM mode.")
    return MockLLM()


class MockLLM:
    """A very simple mock LLM that uses the retrieved contexts to build an answer."""

    def __call__(self, prompt: str) -> str:
        # Heuristic: find the "Context:" section and echo a summarized version.
        if "Context:" in prompt:
            ctx = prompt.split("Context:")[1].split("Question:")[0].strip()
            # Shorten and present as a bulleted list
            lines = [line.strip() for line in ctx.splitlines() if line.strip()]
            lines = lines[:5]
            summary = " | ".join([textwrap.shorten(l, width=140) for l in lines])
            return f"(MOCK) Based on the retrieved FAQs: {summary}\n\nIf this doesn't answer your question, please contact support@ourbrand.com."
        return "(MOCK) I don't have enough information."


def load_index(index_dir: str, embeddings: Embeddings):
    if FAISS is None:
        raise RuntimeError("Required packages are missing. Please install requirements.txt first.")

    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index directory '{index_dir}' not found. Run ingest.py first.")

    print(f"Loading FAISS index from '{index_dir}' ...")
    vectorstore = FAISS.load_local(folder_path=index_dir, embedding=embeddings)

    # Load stored docs metadata if available
    docs = []
    try:
        with open(DOCS_PICKLE, "rb") as f:
            docs = pickle.load(f)
    except Exception:
        pass

    return vectorstore, docs


def build_prompt(retrieved_texts: List[str], question: str) -> str:
    """Build a simple prompt that provides context to the LLM."""
    context = "\n\n".join(retrieved_texts)
    prompt = f"""You are a helpful customer support assistant for a D2C brand. Use the following context from the brand's FAQ to answer the question. If the answer is not contained in the context, say you don't know and give a concise next step.

Context:
{context}

Question: {question}
Answer (concise):
"""
    return prompt


def run_chatbot(mock: bool, index_dir: str = INDEX_DIR):
    embeddings = get_embeddings(mock=mock)
    vectorstore, docs = load_index(index_dir, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = get_llm(mock=mock)

    print("\nRAG FAQ Chatbot — type a question and press enter. Type 'exit' or Ctrl+C to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # Retrieve relevant FAQ chunks
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_texts = [d.page_content for d in retrieved_docs]

        if not retrieved_texts:
            print("No relevant FAQ entries found. Try rephrasing your question or contact support.")
            continue

        # Show short preview of retrieved contents (helpful for debugging / transparency)
        print("\nRetrieved (top results):")
        for i, doc in enumerate(retrieved_docs, start=1):
            preview = textwrap.shorten(doc.page_content.replace("\n", " "), width=160)
            print(f"  {i}. {preview}")

        # Build prompt and call LLM
        prompt = build_prompt(retrieved_texts, query)
        try:
            answer = llm(prompt)
        except Exception as e:
            answer = f"LLM error: {e}"

        print("\nAssistant:", answer)
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Start the RAG FAQ chatbot.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no API calls).")
    parser.add_argument("--index-dir", default=INDEX_DIR, help="Directory where FAISS index is saved.")
    args = parser.parse_args()

    run_chatbot(mock=args.mock, index_dir=args.index_dir)


if __name__ == "__main__":
    main()

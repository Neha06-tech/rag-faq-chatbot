# RAG FAQ Chatbot (Beginner-friendly)

A small Python project demonstrating a Retrieval-Augmented Generation (RAG) FAQ chatbot for direct-to-consumer (D2C) brands.

This project shows how to:
- Ingest a local FAQ file into a vector index (FAISS)
- Retrieve relevant FAQ entries for a user question
- Use an LLM to generate an answer using retrieved context
- Run in a mock/offline mode so you can try the system without API keys

---

## What is RAG? (Simple explanation)

RAG stands for Retrieval-Augmented Generation. Instead of asking an LLM to generate answers purely from its parameters, RAG first retrieves relevant documents (from your knowledge base) and then provides those documents as context to the LLM. This helps:

- Provide factual, up-to-date answers grounded in your data
- Reduce hallucinations (answers not based on real information)
- Make it easy to add or change knowledge without re-training the model

For a D2C brand, RAG helps customer support by answering product, shipping, return, and policy questions directly from the brand's own FAQ content.

---

## Problem it solves for D2C customer support

- Customers often ask the same questions repeatedly (returns, shipping, materials).
- Support teams spend time answering repetitive questions.
- With RAG, a chatbot can answer common questions instantly with answers pulled directly from brand content, freeing human agents for more complex tasks.

---

## Project Structure

- data/faqs.txt          — sample FAQ content (plain text)
- ingest.py              — reads `faqs.txt`, creates embeddings, and builds a FAISS index
- chatbot.py             — loads the FAISS index and runs an interactive chatbot
- requirements.txt       — Python dependencies
- README.md              — this file

---

## How the system works (step-by-step)

1. Ingest:
   - `ingest.py` reads `data/faqs.txt`. Each FAQ entry is separated by a blank line.
   - Text entries are converted into embeddings (vector representations).
   - FAISS builds a vector index from the embeddings and saves it to disk.

2. Query / Retrieval:
   - `chatbot.py` loads the FAISS index and turns it into a retriever.
   - When a user asks a question, the retriever finds the top-k FAQ entries most similar to the question.

3. Generation:
   - The retrieved FAQ text chunks are combined into a prompt and sent to an LLM (OpenAI or HuggingFaceHub).
   - The LLM generates a concise answer using the provided context.

4. Mock mode:
   - If you don't have API keys (OpenAI/HuggingFace), both the embedding and LLM parts can run in a mock mode so you can experiment locally.

---

## Tools & Technologies Used

- Python (simple scripts)
- LangChain — orchestration (Documents, vector store helpers, LLM wrappers)
- FAISS — efficient vector index and similarity search
- sentence-transformers (HuggingFace) — embedding model (recommended)
- OpenAI or HuggingFaceHub — LLM provider (optional)
- Mock/deterministic fallback — works offline without API keys

---

## Quickstart — Run locally

1. Clone the repo (or copy files into a folder).

2. Create a Python environment and install dependencies:

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows

   pip install -r requirements.txt

3. Ingest the FAQs and build the FAISS index.

   - Offline / no API keys (mock mode):
     python ingest.py --mock

   - With HuggingFace sentence-transformers (recommended, no API key needed):
     python ingest.py

     (This uses the `sentence-transformers/all-MiniLM-L6-v2` model to create embeddings. It will download the model the first time.)

   - With OpenAI embeddings (optional; requires OPENAI_API_KEY):
     You could adapt the code to use OpenAI embeddings, but this example uses HuggingFace or mock.

4. Run the chatbot:

   - Mock mode (no API keys):
     python chatbot.py --mock

   - Try using OpenAI (if you have an API key):
     export OPENAI_API_KEY="sk-..."
     python chatbot.py

   - Or use HuggingFaceHub (if you have a token):
     export HUGGINGFACEHUB_API_TOKEN="hf_..."
     python chatbot.py

5. Interact:
   - Type a question (e.g., "How can I return an item?") and press Enter.
   - Type `exit` to quit.

---

## Example questions & example outputs

Example 1
- You: How can I return an item?
- Assistant: (LLM returns a concise answer like)
  "You can return unopened products within 30 days for a full refund. Start a return via our Returns portal or contact support@ourbrand.com."

Example 2
- You: Do you ship internationally?
- Assistant: "Yes — we ship to many countries. Shipping costs and delivery times vary by destination. Customers are responsible for customs duties."

Example 3 (when the FAQ doesn't contain the answer)
- You: Do you have a physical store in NYC?
- Assistant: "I don't see this in the FAQ. Please contact support@ourbrand.com for store location information."

When running in mock mode, responses will be clearly marked as mock and will be generated deterministically from the retrieved text.

---

## Notes & Tips

- The FAQ file is intentionally simple. Each FAQ entry is separated by a blank line.
- The mock embeddings are deterministic but not semantically meaningful — they are only for demo purposes.
- For production, use a proper embedding model (sentence-transformers or OpenAI embeddings) and a reliable LLM (OpenAI, hosted HF model, or an internal model) with proper rate limits and monitoring.
- You can expand the system by:
  - Adding chunking (split long documents into smaller passages)
  - Storing metadata (URLs, product IDs) and returning sources with answers
  - Using a proper prompt template or a retrieval-augmented chain from LangChain

---

If something doesn't work (missing packages, errors), double-check that your virtual environment is active and `pip install -r requirements.txt` completed successfully.

Enjoy experimenting with RAG for customer support!

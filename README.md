# FAISS + Gemini RAG Demo

A modular Retrieval-Augmented Generation (RAG) pipeline built with PyMuPDF, Google Gemini embeddings & generation, and FAISS vector search.

---

## Run Commands

```powershell
cd D:\Sudarchanan\git-clones\faiss-rag-demo

python main.py extract
python main.py chunk
python main.py embed
python main.py index
python main.py search "What is the main topic of this document?"
python main.py retrieve "What is the main topic of this document?"
python main.py query "What is the main topic of this document?"
python main.py evaluate --limit 2
```

---

## Pipeline Overview

1. **Extract**: Extracts per-page text from PDFs in `data/documents/` into `output/<name>.json`.
   ```powershell
   python main.py extract
   ```

2. **Chunk**: Slices extracted document text into inspectable word chunks with continuous overlap into `output/<name>.chunks.json`.
   ```powershell
   python main.py chunk
   ```

3. **Embed**: Generates dense vector embeddings via Gemini Embedding 2 and saves to `output/<name>.embeddings.json`.
   ```powershell
   python main.py embed
   ```

4. **Index**: Builds and persists the FAISS inner-product vector index under `storage/index.faiss`.
   ```powershell
   python main.py index
   ```

5. **Search**: Performs vector similarity search against the FAISS index and prints retrieved chunks with similarity scores and page numbers.
   ```powershell
   python main.py search "What is the main topic of this document?"
   ```

6. **Retrieve**: Assembles structured context and citation metadata across retrieved chunks without calling the LLM generator.
   ```powershell
   python main.py retrieve "What is the main topic of this document?"
   ```

7. **Query**: End-to-end RAG that retrieves relevant context and generates answers using Gemini.
   ```powershell
   python main.py query "What happened during the Sol 94 dust storm?"
   ```

8. **Evaluate**: Evaluates RAG quality against the golden test set using RAGAS metrics (Faithfulness, Context Recall, Context Precision, Answer Relevancy).
   ```powershell
   # Quick smoke test with 2 questions
   python main.py evaluate --limit 2

   # Full evaluation on all questions using Gemma 4 26B judge
   python main.py evaluate --judge-model gemma-4-26b-a4b-it

   # Re-evaluate cached answers without re-running generation
   python main.py evaluate --reuse-samples output/eval/samples_<timestamp>.json
   ```

# RAG Ingestion Pipeline Issues & Improvements

This document tracks identified issues, resolved bugs, and recommended optimizations for the document extraction and chunking pipeline in `faiss-rag-demo`.

---

## 1. Resolved Issues

### 1.1 Non-Sequential and Duplicate `chunk_number`s
- **Problem**: In previous iterations, `list.extend()` was passed a generator expression referencing `len(chunks) + index + 1`. Because `extend()` mutates the list as items are yielded, `len(chunks)` increased dynamically while `index` also increased, producing skipping chunk numbers (e.g., `1, 2, 4, 4, 6, 6...`) and collisions on subsequent pages.
- **Status**: **Resolved**. The pipeline in `rag/ingestion/chunk_pipeline.py` now sequences global chunks consecutively (`1` through `N`) using `enumerate(..., start=1)`.

### 1.2 Isolated Page Chunking & Loss of Cross-Page Overlap
- **Problem**: Chunking was initially run per page in isolation, causing zero overlap across page transitions, inconsistent tail chunk sizes (e.g., 60-word redundant tail chunks), and page-relative `start_word` resets to 0.
- **Status**: **Resolved**. The document is now assembled globally across pages before slicing, preserving consistent overlap (80 words) and tracking `page_start` and `page_end` per chunk.

---

## 2. Active Issues & Areas for Improvement

### Issue 1: Running Headers & Page Footers Embedded in Chunks
- **Location**: Extracted text from `PyMuPDF` (`fitz.open()`) in `rag/ingestion/extractor.py` and downstream chunks.
- **Description**: Running headers such as `"The Ares Dawn Chronicle - A Work of Fiction Page X Chapter Y..."` appear repeatedly at every page transition within chunk text.
- **Impact**:
  - Wastes embedding and LLM prompt tokens.
  - Can distort dense vector retrieval scores by over-weighting repeated header keywords.
- **Proposed Fix**:
  - Filter out repeated header/footer patterns using regex or PyMuPDF bounding-box/block filtering (e.g., ignoring text blocks at the top/bottom margins of each page).

---

### Issue 2: Loss of Structural Formatting and Table Layout
- **Location**: `rag/ingestion/chunk_pipeline.py` (line 16: `page_words = str(page["text"]).split()`).
- **Description**: Using `.split()` and `" ".join(...)` flattens all newline characters (`\n`), indentations, and tabular structures into a continuous single-line string of space-separated words.
- **Impact**:
  - Tables (such as the *Key Facts at a Glance* table on Page 1) lose their row/column associations and become continuous sentences.
  - LLMs have difficulty discerning column keys from their values when context is retrieved.
- **Proposed Fix**:
  - Preserve newline structure during extraction and chunking, or use a markdown-aware / table-preserving extraction strategy.

---

### Issue 3: Hard Word-Boundary Truncation vs. Sentence/Semantic Boundaries
- **Location**: `rag/ingestion/chunk_pipeline.py` (word-based slicing).
- **Description**: Chunks are sliced strictly on word count intervals (`step_size_words = 420`, `chunk_size_words = 500`), frequently splitting sentences mid-clause (e.g., *"...roughly double that of"*).
- **Impact**:
  - Incomplete thoughts at the start and end of chunks degrade retrieval precision and LLM answer generation fidelity.
- **Proposed Fix**:
  - Implement sentence-aware or paragraph-aware boundary snapping (e.g., snapping the chunk boundary to the nearest terminal punctuation: `.`, `?`, `!`, or newline) within a tolerance window around the target chunk size.

---

### Issue 4: Missing `source` Field on Individual Chunk Objects
- **Location**: Chunks list in `output/*.chunks.json`.
- **Description**: The top-level document has `"source": "ares_dawn.pdf"`, but individual objects in the `"chunks"` array do not include `"source"`.
- **Impact**:
  - When chunks are indexed or passed into FAISS / Gemini independently, the source reference must be manually re-attached.
- **Proposed Fix**:
  - Include `"source": source` directly in each chunk object.

---

## 3. Summary Checklist

- [x] Fix duplicate and skipped `chunk_number` values.
- [x] Maintain continuous 80-word overlap across page boundaries.
- [x] Add multi-page tracking (`page_start`, `page_end`).
- [ ] Add header/footer noise removal in text extraction.
- [ ] Preserve line breaks / table structures instead of raw `.split()`.
- [ ] Implement sentence-boundary or recursive character splitting.
- [ ] Include `"source"` field inside individual chunk dictionaries.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Always read `plan.md` at session start to know current phase and next step — it's the living
record of progress across sessions, so check it before assuming what's done vs. still open.

## Commands

Run all commands from this directory (`rag_agents_cover_letter/`) — `pytest.ini` sets `pythonpath = .` so `src` resolves as a package from here.

```bash
# Activate the existing venv (or create one: python -m venv .venv)
source .venv/bin/activate
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test
pytest tests/test_ingest.py::test_load_pdf_documents_reads_pdf_with_cover_letter_metadata -v

# Run one test file
pytest tests/test_retrieve.py -v

# Full ingest pipeline (real OpenAI calls: gpt-4o-mini metadata extraction per PDF + embeddings — costs money, requires OPENAI_API_KEY in .env)
python -m src.ingest

# CLI query against the persisted vector store
python -m src.retrieve "some query"

# Wipe the vector store to force a clean re-ingest
rm -rf chroma_db
```

Notebook (`notebooks/explore.ipynb`) outputs are stripped via `nbstripout` (declared in `.gitattributes`). After cloning, run once: `pip install nbstripout && nbstripout --install`.

## Architecture

Three-stage pipeline, one module each, meant to be composed by callers (no CLI orchestrator ties them together yet):

**`src/ingest.py`** — load → chunk → embed → persist.
- PDFs live under `data/cover letters/`, `data/projects/`, `data/cv/`; each directory has a thin wrapper (`load_pdf_documents`, `load_project_documents`, `load_cv_documents`) around the shared `_load_pdf_documents_from(directory)`.
- For every PDF, `extract_metadata_llm(text, source)` calls `gpt-4o-mini` in JSON mode to derive `type` (`cv`/`project`/`cover_letter`), `company`, and `skills` (comma-separated tools only). This dict becomes the `Document.metadata`, which `RecursiveCharacterTextSplitter.split_documents()` then propagates onto every chunk — so metadata is per-source-document, not per-chunk-derived.
- `ingest()` chunks with `RecursiveCharacterTextSplitter` (300/50, paragraph → line → sentence separators), embeds with `OpenAIEmbeddings(text-embedding-3-small)`, and persists to a Chroma collection (`cv_documents`) at `chroma_db/`. It calls `.delete_collection()` first — re-running `ingest()` replaces the collection rather than appending duplicate chunks.
- `load_documents()`, `chunk_documents()`, and `infer_type()` are a legacy token-count-based path over `data/*.txt` with filename-keyword type inference. They're no longer wired into `ingest()` (see the commented-out line) but are still covered by tests — don't assume they're dead code without checking call sites first.

**`src/retrieve.py`** — query the persisted collection.
- `get_retriever(k, doc_type=None)` builds a Chroma similarity retriever; passing `doc_type` adds a metadata `filter={"type": doc_type}`.
- `retrieve(query, k)` is unfiltered top-k; `retrieve_by_type(query, doc_type, k)` filters by the `type` field. There is no equivalent filter for `company` yet, and any chunks embedded before `extract_metadata_llm` existed won't have a `company` field at all.

**`src/generator.py`** — `generate(context, job_description)` sends retrieved context + a job description through a fixed prompt to `gpt-4o-mini` (temperature 0.3) and returns one cover-letter paragraph. The prompt explicitly forbids inventing facts not present in the supplied context.

## Testing convention

Every test mocks the OpenAI/Chroma boundary (`@patch("src.retrieve.Chroma")`, `@patch("src.generator.ChatOpenAI")`, monkeypatching `ingest.extract_metadata_llm`, etc.) — no test should make a real network call. `tests/test_ingest.py::make_pdf_bytes` builds a minimal single-page PDF in memory byte-by-byte so PDF-loading tests don't need binary fixture files on disk.

## Roadmap

`plan.md` at the repo root is the living master plan (current phase, active node, parking-lot items) — check it before starting work to see what's actually in progress versus already decided against. As of this writing it tracks a move from the RAG core (done) into an agentic LangGraph layer (Tavily company research, gap analysis) with RAGAS evaluation planned after.

`README.md`'s "Improvements" section lists known-but-unimplemented ideas: CV-aware chunking (respecting project/role/skill boundaries instead of raw character counts), hybrid semantic+BM25 search, and reranking top-k before generation.

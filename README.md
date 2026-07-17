# rag_agents_cover_letter
A small RAG pipeline over your own CV, project write-ups, and past cover letters (PDFs under
`data/`). It extracts metadata per document, chunks and embeds them into a persistent Chroma
store, retrieves relevant chunks for a job description, and drafts a tailored cover letter
paragraph from them.

See `CLAUDE.md` for setup/run commands and module-by-module architecture, and `plan.md` for
the current roadmap phase.

## Delete vector store
rm -rf chroma_db

## Metadata extraction
Each PDF is sent to `gpt-4o-mini` once (`ingest.extract_metadata_llm`) to derive `type`
(`cv`/`project`/`cover_letter`), `company` (if the document is addressed to one), and `skills`
(comma-separated tools only). This metadata is attached to the source document and propagates
to every chunk produced from it, so it costs one extra chat completion per PDF on top of the
embedding cost each time you re-run `ingest()`.

## Retrieval filtering
`retrieve.retrieve_by_type(query, doc_type, k)` filters by the `type` field above (e.g. only
`cv` chunks, or only `cover_letter` chunks) instead of searching the whole collection. There's
no equivalent `company` filter yet, and any chunks embedded before metadata extraction existed
won't have a `company` field until you re-ingest.

## Chunking
`ingest()` currently chunks with `RecursiveCharacterTextSplitter` (chunk_size=300,
chunk_overlap=50, splitting on paragraph → line → sentence boundaries first). Most of the
ingested content is past cover letters, which don't have a fixed structure, so a fairly large,
generic chunk size works better than trying to split by section headers. There's also a legacy
token-based chunker (`chunk_documents`, 500 tokens / 50 overlap) kept for reference — it isn't
called by `ingest()` anymore, but is still covered by tests and used in the notebook's `.txt`
exploration section.

## Notebook outputs
Notebook outputs are stripped from git via `nbstripout`, so `.ipynb` diffs only show code/markdown changes, not executed outputs. After cloning, run:
```
pip install nbstripout
nbstripout --install
```
This registers a git filter in your local `.git/config` — it needs to be run once per clone, since `.gitattributes` only declares *what* to filter, not *how*.

## Improvments
1. Better chunking for a CV - RecursiveCharacterTextSplitter splits by character count, which is dumb for a CV. It might cut a sentence mid-thought. A CV has natural boundaries: projects, roles, skills. Respect them.
2. Hybrid search
Right now you only use semantic similarity (embedding distance). The problem: if a job says "Factorization Machines" and your CV says "Factorization Machines", pure semantic search might miss it because the exact term gets averaged out in the embedding. Hybrid search combines semantic + keyword (BM25) so exact technical terms get matched too.
Chroma doesn't support this natively but it's worth knowing for the interview story. Weaviate does it out of the box.
3. Reranking
The top 4 chunks by cosine similarity are not necessarily the 4 most useful for generation. A reranker (Cohere Rerank API, or cross-encoder/ms-marco-MiniLM from HuggingFace) re-scores the top-k retrieved chunks with a more expensive but accurate model.
The pattern: retrieve top 10, rerank, pass top 4 to the LLM. Better signal, same generation cost.
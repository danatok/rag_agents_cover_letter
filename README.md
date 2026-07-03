# rag_agents_cover_letter
This code helps to read your repo with txt files and write you a new cover letter

## Delete vector store
rm -rf chroma_db

## Chunking
Chunk size is currently set to 500 tokens (50 token overlap). Most of the ingested content is past cover letters, which don't have a fixed structure, so a fairly large, generic chunk size works better than trying to split by section headers.

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
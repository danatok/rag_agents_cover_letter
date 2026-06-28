# rag_agents_cover_letter
This code helps to read your repo with txt files and write you a new cover letter

## Chunking
Chunk size is currently set to 500 tokens (50 token overlap). Most of the ingested content is past cover letters, which don't have a fixed structure, so a fairly large, generic chunk size works better than trying to split by section headers.

## Notebook outputs
Notebook outputs are stripped from git via `nbstripout`, so `.ipynb` diffs only show code/markdown changes, not executed outputs. After cloning, run:
```
pip install nbstripout
nbstripout --install
```
This registers a git filter in your local `.git/config` — it needs to be run once per clone, since `.gitattributes` only declares *what* to filter, not *how*.


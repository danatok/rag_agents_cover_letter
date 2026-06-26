"""Chunk the .txt files in data/, embed them, and store in a persistent Chroma collection."""

from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

load_dotenv()

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
ENCODING = tiktoken.get_encoding("cl100k_base")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COVER_LETTERS_DIR = DATA_DIR / "cover letters"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "cv_documents"

# Per-file metadata. Edit this as you add/rename files in data/.
# "skills" is a free-form list you can use to filter retrieval later.
FILE_METADATA = {
    "cv.txt": {"type": "cv", "skills": []},
    "cover_letter.txt": {"type": "cover_letter", "skills": []},
    "cover_letter_old.txt": {"type": "cover_letter", "skills": []},
    "cover_letter_general.txt": {"type": "cover_letter", "skills": []},
    "cover_letter sonova.txt": {"type": "cover_letter", "skills": []},
    "project_recommender.txt": {"type": "project", "skills": []},
    "model monitoring.txt": {"type": "project", "skills": []},
    "motivational.txt": {"type": "other", "skills": []},
}


def load_documents() -> list[Document]:
    documents = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        meta = FILE_METADATA.get(path.name, {"type": "other", "skills": []})
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "type": meta["type"],
                    "skills": ", ".join(meta["skills"]),
                },
            )
        )
    return documents


def load_pdf_documents() -> list[Document]:
    documents = []
    for path in sorted(COVER_LETTERS_DIR.glob("*.pdf")):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "type": "cover_letter",
                    "skills": "",
                },
            )
        )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = []
    for doc in documents:
        tokens = ENCODING.encode(doc.page_content)
        step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
        for start in range(0, len(tokens), step):
            window = tokens[start : start + CHUNK_SIZE_TOKENS]
            if not window:
                continue
            chunks.append(
                Document(
                    page_content=ENCODING.decode(window),
                    metadata=doc.metadata,
                )
            )
            if start + CHUNK_SIZE_TOKENS >= len(tokens):
                break
    return chunks


def ingest() -> Chroma:
    documents = load_documents() + load_pdf_documents()
    chunks = chunk_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Ingested {len(documents)} files into {len(chunks)} chunks -> {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    ingest()

"""Chunk the .txt files in data/, embed them, and store in a persistent Chroma collection."""

from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
ENCODING = tiktoken.get_encoding("cl100k_base")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COVER_LETTERS_DIR = DATA_DIR / "cover letters"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "cv_documents"

# Keyword -> type, checked in order against the lowercased filename.
TYPE_KEYWORDS = [
    ("cover_letter", "cover_letter"),
    ("cover letter", "cover_letter"),
    ("cv", "cv"),
    ("project", "project"),
    ("model", "project"),
]


def infer_type(filename: str) -> str:
    name = filename.lower()
    for keyword, doc_type in TYPE_KEYWORDS:
        if keyword in name:
            return doc_type
    return "other"


def load_documents() -> list[Document]:
    documents = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "type": infer_type(path.name),
                    "skills": "",
                },
            )
        )
    return documents


def load_pdf_documents() -> list[Document]:
    documents = []
    for path in sorted(COVER_LETTERS_DIR.glob("*.pdf")):
        pages = PyPDFLoader(str(path)).load()
        text = "\n".join(page.page_content for page in pages).strip()
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
    #documents = load_documents() + load_pdf_documents()
    documents = load_pdf_documents()

    # instead of splitting by characters, split by your own delimiters
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=300,
    #     chunk_overlap=50,
    #     separators=["\n\n", "\n", ". "]  # try paragraph breaks first, then lines, then sentences
    # )
    # chunks = splitter.split_documents(documents)
    chunks = chunk_documents(documents)
    # Development / portfolio project → OpenAI is fine
    # Clean, simple, no truncation worry
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # If you want fully local + no truncation worry
    #embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    # 512 token limit, free, best open-source retrieval quality
    with get_openai_callback() as cb:
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
        )
    print(f"Ingested {len(documents)} files into {len(chunks)} chunks -> {CHROMA_DIR}")
    print(f"Tokens used: {cb.total_tokens} | Cost: ${cb.total_cost:.4f}")
    return vectorstore


if __name__ == "__main__":
    ingest()

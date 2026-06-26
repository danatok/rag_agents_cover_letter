"""Query the persisted Chroma collection and return the top-k matching chunks."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.ingest import CHROMA_DIR, COLLECTION_NAME

load_dotenv()

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return _vectorstore


def retrieve(query: str, k: int = 4) -> list[Document]:
    return get_vectorstore().similarity_search(query, k=k)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "machine learning engineer experience"
    for doc in retrieve(query):
        print(f"[{doc.metadata.get('source')}] {doc.page_content[:120]}...")

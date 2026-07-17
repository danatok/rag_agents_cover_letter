"""Query the persisted Chroma collection and return the top-k matching chunks."""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.ingest import CHROMA_DIR, COLLECTION_NAME

load_dotenv()


def get_retriever(k: int = 4, doc_type: str | None = None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    search_kwargs = {"k": k}
    if doc_type is not None:
        search_kwargs["filter"] = {"type": doc_type}
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def retrieve(query: str, k: int = 4) -> list[str]:
    chunks = get_retriever(k).invoke(query)
    return [c.page_content for c in chunks]


def retrieve_by_type(query: str, doc_type: str, k: int = 4) -> list[str]:
    chunks = get_retriever(k, doc_type=doc_type).invoke(query)
    return [c.page_content for c in chunks]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "machine learning engineer experience"
    for chunk in retrieve(query):
        print(chunk[:200])
        print("---")

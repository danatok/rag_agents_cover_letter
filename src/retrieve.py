"""Query the persisted Chroma collection and return the top-k matching chunks."""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.ingest import CHROMA_DIR, COLLECTION_NAME

load_dotenv()


def get_retriever(k: int = 4):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve(query: str, k: int = 4) -> list[str]:
    chunks = get_retriever(k).invoke(query)
    return [c.page_content for c in chunks]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "machine learning engineer experience"
    for chunk in retrieve(query):
        print(chunk[:200])
        print("---")

"""Integration check: does the metadata schema ingest.py produces actually work
end-to-end through chunking and retrieve.py's real (non-mocked) filtering logic?

Uses a deterministic, offline embedding stand-in so this stays a fast, free test
rather than a real OpenAI call - only the metadata/filter contract is under test.
"""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import ingest, retrieve

METADATA_KEYS = {"source", "type", "company", "skills"}


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings - no network calls."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * 8
        for i, ch in enumerate(text):
            vec[i % 8] += ord(ch)
        return vec


def _make_raw_docs() -> list[ingest.Document]:
    # Mirrors the real dict shape extract_metadata_llm returns (ingest.py:43-48),
    # without calling gpt-4o-mini.
    return [
        ingest.Document(
            page_content="Experienced ML engineer building production systems. " * 30,
            metadata={"source": "cv.pdf", "type": "cv", "company": "", "skills": "Python, PyTorch"},
        ),
        ingest.Document(
            page_content="Dear Hiring Team at Acme, I am excited to apply for this role. " * 30,
            metadata={"source": "letter_acme.pdf", "type": "cover_letter", "company": "Acme", "skills": ""},
        ),
        ingest.Document(
            page_content="Built a recommender system using factorization machines. " * 30,
            metadata={"source": "project.pdf", "type": "project", "company": "", "skills": "PySpark, SQL"},
        ),
    ]


def test_metadata_survives_chunking_and_type_filter_returns_correct_chunks(tmp_path, monkeypatch):
    raw_docs = _make_raw_docs()

    # Chunk exactly the way ingest() does (ingest.py:148-154), to prove metadata
    # propagates onto every resulting chunk, not just the source document.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". "]
    )
    chunks = splitter.split_documents(raw_docs)
    assert len(chunks) > len(raw_docs), "expected splitting to produce multiple chunks per doc"
    for chunk in chunks:
        assert set(chunk.metadata) == METADATA_KEYS

    # Point retrieve.py at a throwaway local Chroma collection with fake embeddings,
    # then exercise its real (unmocked) get_retriever/retrieve_by_type code paths.
    monkeypatch.setattr(retrieve, "OpenAIEmbeddings", lambda **kwargs: _FakeEmbeddings())
    monkeypatch.setattr(retrieve, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(retrieve, "COLLECTION_NAME", "test_collection")

    Chroma.from_documents(
        chunks,
        embedding=_FakeEmbeddings(),
        collection_name="test_collection",
        persist_directory=str(tmp_path),
    )

    for doc_type in ("cv", "cover_letter", "project"):
        docs = retrieve.get_retriever(k=5, doc_type=doc_type).invoke("relevant experience")
        assert docs, f"filter for type={doc_type!r} returned nothing"
        assert all(d.metadata["type"] == doc_type for d in docs), (
            f"retrieve_by_type leaked a chunk of a different type for {doc_type!r}"
        )

        results = retrieve.retrieve_by_type("relevant experience", doc_type, k=5)
        assert results, f"retrieve_by_type returned nothing for type={doc_type!r}"
        assert all(isinstance(r, str) for r in results)

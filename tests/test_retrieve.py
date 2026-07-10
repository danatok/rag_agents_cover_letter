from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src import retrieve


@patch("src.retrieve.Chroma")
@patch("src.retrieve.OpenAIEmbeddings")
def test_get_retriever_uses_correct_collection_and_dir(mock_embeddings, mock_chroma):
    retrieve.get_retriever(k=4)

    mock_chroma.assert_called_once_with(
        persist_directory=str(retrieve.CHROMA_DIR),
        embedding_function=mock_embeddings.return_value,
        collection_name=retrieve.COLLECTION_NAME,
    )


@patch("src.retrieve.Chroma")
@patch("src.retrieve.OpenAIEmbeddings")
def test_get_retriever_passes_k_to_as_retriever(mock_embeddings, mock_chroma):
    retrieve.get_retriever(k=7)

    mock_chroma.return_value.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 7},
    )


@patch("src.retrieve.get_retriever")
def test_retrieve_returns_page_content_strings(mock_get_retriever):
    mock_get_retriever.return_value.invoke.return_value = [
        Document(page_content="chunk one", metadata={}),
        Document(page_content="chunk two", metadata={}),
    ]

    result = retrieve.retrieve("some query", k=2)

    assert result == ["chunk one", "chunk two"]
    mock_get_retriever.return_value.invoke.assert_called_once_with("some query")


@patch("src.retrieve.get_retriever")
def test_retrieve_returns_empty_list_when_no_chunks(mock_get_retriever):
    mock_get_retriever.return_value.invoke.return_value = []

    result = retrieve.retrieve("obscure query")

    assert result == []

from unittest.mock import MagicMock, patch

from src import generator


@patch("src.generator.ChatOpenAI")
def test_generate_returns_response_content(mock_llm_cls):
    mock_response = MagicMock()
    mock_response.content = "Dana's ML background is a strong fit."

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch.object(generator, "_PROMPT", mock_prompt):
        result = generator.generate(
            context="Built ML pipelines at ABN AMRO.",
            job_description="Seeking a senior ML engineer.",
        )

    assert result == "Dana's ML background is a strong fit."
    mock_chain.invoke.assert_called_once_with({
        "context": "Built ML pipelines at ABN AMRO.",
        "job_description": "Seeking a senior ML engineer.",
    })


@patch("src.generator.ChatOpenAI")
def test_generate_uses_correct_model_and_temperature(mock_llm_cls):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="result")

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch.object(generator, "_PROMPT", mock_prompt):
        generator.generate("ctx", "jd")

    mock_llm_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0.3)

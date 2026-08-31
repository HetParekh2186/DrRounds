"""Tests for doctor_rounds.adapters.llm.

Each adapter wraps a real external call (a local Ollama server, or a
hosted API) — these tests mock the underlying client/HTTP call so they run
fast and offline, while still exercising the adapter's own logic: how the
request is built and how the response is parsed back into plain text.
"""

from unittest.mock import MagicMock, patch

from doctor_rounds.adapters.llm import AnthropicLLM, OllamaLLM, OpenAILLM


class TestOllamaLLM:
    def test_sends_expected_request_and_parses_response(self):
        llm = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")
        fake_response = MagicMock()
        fake_response.json.return_value = {"response": "Metformin is first-line therapy."}
        fake_response.raise_for_status.return_value = None

        with patch("httpx.post", return_value=fake_response) as mock_post:
            result = llm.generate("What's first-line for type 2 diabetes?")

        assert result == "Metformin is first-line therapy."
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "llama3.1"
        assert kwargs["json"]["prompt"] == "What's first-line for type 2 diabetes?"
        assert kwargs["json"]["stream"] is False

    def test_strips_trailing_slash_from_base_url(self):
        llm = OllamaLLM(base_url="http://localhost:11434/")
        assert llm.base_url == "http://localhost:11434"

    def test_raises_on_http_error(self):
        import httpx

        llm = OllamaLLM()
        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=MagicMock()
        )
        with patch("httpx.post", return_value=fake_response):
            try:
                llm.generate("test")
                raised = False
            except httpx.HTTPStatusError:
                raised = True
        assert raised


class TestAnthropicLLM:
    def test_extracts_text_from_response_content_blocks(self):
        llm = AnthropicLLM(model="claude-sonnet-5", api_key="fake-key")
        text_block = MagicMock(type="text", text="Aspirin inhibits platelet aggregation.")
        fake_message = MagicMock(content=[text_block])
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_message

        with patch("anthropic.Anthropic", return_value=fake_client) as mock_ctor:
            result = llm.generate("How does aspirin work?")

        assert result == "Aspirin inhibits platelet aggregation."
        mock_ctor.assert_called_once_with(api_key="fake-key")
        _, kwargs = fake_client.messages.create.call_args
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["messages"] == [{"role": "user", "content": "How does aspirin work?"}]

    def test_joins_multiple_text_blocks_and_skips_non_text(self):
        llm = AnthropicLLM(api_key="fake-key")
        blocks = [
            MagicMock(type="text", text="Part one. "),
            MagicMock(type="tool_use", text="ignored"),
            MagicMock(type="text", text="Part two."),
        ]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = MagicMock(content=blocks)

        with patch("anthropic.Anthropic", return_value=fake_client):
            result = llm.generate("prompt")

        assert result == "Part one. Part two."


class TestOpenAILLM:
    def test_extracts_message_content(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="fake-key")
        fake_choice = MagicMock()
        fake_choice.message.content = "Beta blockers reduce heart rate."
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(choices=[fake_choice])

        with patch("openai.OpenAI", return_value=fake_client) as mock_ctor:
            result = llm.generate("What do beta blockers do?")

        assert result == "Beta blockers reduce heart rate."
        mock_ctor.assert_called_once_with(api_key="fake-key")

    def test_returns_empty_string_when_content_is_none(self):
        # a real OpenAI response can have null content (e.g. a refusal) —
        # generate() should degrade to "" rather than raise
        llm = OpenAILLM(api_key="fake-key")
        fake_choice = MagicMock()
        fake_choice.message.content = None
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(choices=[fake_choice])

        with patch("openai.OpenAI", return_value=fake_client):
            result = llm.generate("prompt")

        assert result == ""

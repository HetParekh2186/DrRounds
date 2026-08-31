"""LLM adapters — anything that can turn a prompt into text.

`LLM` is a `Protocol`, not a base class: evaluating *your* pipeline means
accepting whatever object you already generate answers with, as long as it
exposes a `generate(prompt: str) -> str` method — including ones this
project has never heard of. The three implementations below exist so
there's always a zero-setup way to try the framework end to end.

`OllamaLLM` is the recommended default for anyone cloning this repo: it
talks to a locally running model, so running the example pipeline costs
nothing and needs no API key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaLLM:
    """A locally running model served by Ollama (https://ollama.com).

    No API key, no per-token cost, works offline once the model is pulled
    (`ollama pull llama3.1`) — the default way to run Doctor Rounds'
    examples without any account setup.
    """

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        result: str = response.json()["response"]
        return result


class AnthropicLLM:
    """Uses the Anthropic API. Requires `pip install doctor-rounds[anthropic]`
    and an `ANTHROPIC_API_KEY` (or an explicit `api_key`)."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


class OpenAILLM:
    """Uses the OpenAI API. Requires `pip install doctor-rounds[openai]`
    and an `OPENAI_API_KEY` (or an explicit `api_key`)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

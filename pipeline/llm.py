"""
LLM client abstraction — routes through OpenRouter (OpenAI-compatible API).

All pipeline LLM calls go through this module. To switch providers, change
the base_url and api_key logic here.

Environment variable: OPENROUTER_API_KEY
"""

import os

from openai import AsyncOpenAI, OpenAI


def get_client() -> OpenAI:
    """Create a synchronous LLM client."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )


def get_async_client() -> AsyncOpenAI:
    """Create an asynchronous LLM client."""
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )


def chat(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    """Synchronous chat completion. Returns the response text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def achat(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> str:
    """Asynchronous chat completion. Returns the response text."""
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

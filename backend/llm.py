"""LLM integration using Groq API."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from groq import AsyncGroq

from backend.config import get_settings
from backend.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a research assistant. Answer the question based only on the provided context from research papers. Always cite which part of the context you used. Keep answers concise."""


@dataclass
class LLMResponse:
    """Response from the LLM."""

    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} - Page {chunk.page_number}]\n{chunk.content}"
        )
    return "\n\n".join(context_parts)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the full prompt with context."""
    context = build_context(chunks)
    return f"""Context:
{context}

Question: {query}

Answer:"""


def _client() -> AsyncGroq:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured")
    return AsyncGroq(api_key=settings.groq_api_key)


def _messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(query, chunks)},
    ]


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> LLMResponse:
    """Generate a full answer using the Groq API."""
    client = _client()
    logger.info(f"Generating answer for query: '{query[:50]}...'")

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=_messages(query, chunks),
        max_tokens=settings.llm_max_tokens,
        temperature=0.1,
    )

    answer = response.choices[0].message.content or ""
    usage = response.usage
    logger.info(f"Generated answer with {usage.total_tokens} tokens")

    return LLMResponse(
        answer=answer,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        model=settings.llm_model,
    )


async def stream_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> AsyncIterator[str]:
    """Stream answer tokens from Groq."""
    client = _client()
    logger.info(f"Streaming answer for query: '{query[:50]}...'")

    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=_messages(query, chunks),
        max_tokens=settings.llm_max_tokens,
        temperature=0.1,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

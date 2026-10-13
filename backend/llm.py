"""LLM integration using Groq API."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from groq import AsyncGroq

from backend.config import get_settings
from backend.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a research assistant. Answer using only the provided context from research papers.
Cite sources as (Page N), matching the page labels in the context. Never cite as Source 1 or Source 2.
Keep answers concise. If the user asks a follow-up, use the conversation and the context."""


@dataclass
class LLMResponse:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


def build_context(chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Page {chunk.page_number}]\n{chunk.content}")
    return "\n\n".join(context_parts)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context(chunks)
    return f"""Context:
{context}

Question: {query}

Answer:"""


def _client() -> AsyncGroq:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured")
    return AsyncGroq(api_key=settings.groq_api_key)


def _messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:2000]})
    messages.append({"role": "user", "content": build_prompt(query, chunks)})
    return messages


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> LLMResponse:
    client = _client()
    logger.info(f"Generating answer for query: '{query[:50]}...'")

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=_messages(query, chunks, history),
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
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    client = _client()
    logger.info(f"Streaming answer for query: '{query[:50]}...'")

    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=_messages(query, chunks, history),
        max_tokens=settings.llm_max_tokens,
        temperature=0.1,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

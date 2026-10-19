"""Legacy API signatures backed by the shared OpenRouter BYOK SDK client."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from backend.config import get_settings
from backend.openrouter_client import async_client, async_completion, log_model
from backend.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a research assistant. Answer using only the provided context from research papers.
Write in clear prose or a short numbered list. Use simple markdown only when a list helps.
Cite inline as (Page N). Never cite as Source 1 or Source 2. Do not dump a list of pages at the end.
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
    async with async_client() as client:
        response = await async_completion(
            client,
            messages=_messages(query, chunks, history),
            max_tokens=settings.llm_max_tokens,
            temperature=0.1,
        )
    usage = response.usage
    return LLMResponse(
        answer=response.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        model=response.model,
    )


async def stream_answer(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    async with async_client() as client:
        stream = await async_completion(
            client,
            messages=_messages(query, chunks, history),
            max_tokens=settings.llm_max_tokens,
            temperature=0.1,
            stream=True,
            stream_options={"include_usage": True},
        )
        first = True
        try:
            async for chunk in stream:
                if first:
                    log_model(chunk)
                    first = False
                if chunk.choices:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
        finally:
            await stream.close()

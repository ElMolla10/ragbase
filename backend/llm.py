"""LLM integration for Groq or a local Ollama server."""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from groq import AsyncGroq

from backend.config import get_settings
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


def _payload(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None,
    *,
    stream: bool,
) -> dict:
    return {
        "model": settings.llm_model,
        "messages": _messages(query, chunks, history),
        "max_tokens": settings.llm_max_tokens,
        "temperature": 0.1,
        "stream": stream,
    }


def _groq() -> AsyncGroq:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured")
    return AsyncGroq(api_key=settings.groq_api_key)


async def _ollama_create(payload: dict) -> dict:
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise ValueError(
            "Local model is not reachable. Start Ollama and pull the model."
        ) from exc


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> LLMResponse:
    logger.info(f"Generating answer for query: '{query[:50]}...'")

    if settings.llm_provider == "ollama":
        data = await _ollama_create(_payload(query, chunks, history, stream=False))
        answer = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            answer=answer,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=settings.llm_model,
        )

    response = await _groq().chat.completions.create(
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
    logger.info(f"Streaming answer for query: '{query[:50]}...'")

    if settings.llm_provider == "ollama":
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        payload = _payload(query, chunks, history, stream=True)
        try:
            client = httpx.AsyncClient(timeout=180)
            async with client, client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0].get("delta", {}).get("content") or ""
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ValueError(
                "Local model is not reachable. Start Ollama and pull the model."
            ) from exc
        return

    stream = await _groq().chat.completions.create(
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

"""LLM integration using Groq API."""

import logging
from dataclasses import dataclass

from groq import AsyncGroq

from backend.config import get_settings
from backend.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a research assistant. Answer the question based only on the provided context from research papers. Always cite which part of the context you used."""


@dataclass
class LLMResponse:
    """Response from the LLM."""

    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks.

    Args:
        chunks: List of retrieved chunks.

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} - Page {chunk.page_number}]\n{chunk.content}"
        )
    return "\n\n".join(context_parts)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the full prompt with context.

    Args:
        query: User's question.
        chunks: Retrieved context chunks.

    Returns:
        Full prompt string.
    """
    context = build_context(chunks)
    return f"""Context:
{context}

Question: {query}

Answer:"""


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> LLMResponse:
    """Generate an answer using the Groq API.

    Args:
        query: User's question.
        chunks: Retrieved context chunks.

    Returns:
        LLMResponse with answer and token usage.

    Raises:
        ValueError: If no API key is configured.
    """
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured")

    client = AsyncGroq(api_key=settings.groq_api_key)

    prompt = build_prompt(query, chunks)

    logger.info(f"Generating answer for query: '{query[:50]}...'")

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
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

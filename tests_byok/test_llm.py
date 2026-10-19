"""Exercise public generation contracts without loading embedding weights."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from backend.openrouter_client import MODELS, async_client
from tests_byok.test_client import response


@pytest.fixture
def llm(monkeypatch):
    config = ModuleType("backend.config")
    config.get_settings = lambda: SimpleNamespace(llm_max_tokens=512)
    retrieval = ModuleType("backend.retrieval")
    retrieval.RetrievedChunk = SimpleNamespace
    monkeypatch.setitem(sys.modules, "backend.config", config)
    monkeypatch.setitem(sys.modules, "backend.retrieval", retrieval)
    name = "backend._byok_contract_llm"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "backend/llm.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_answer_retains_response_type_history_and_context(llm, monkeypatch):
    captured = []
    monkeypatch.setattr(
        llm,
        "async_client",
        lambda: async_client(
            key="synthetic-not-a-real-key",
            transport=httpx.MockTransport(lambda req: response(req, captured)),
        ),
    )
    result = await llm.generate_answer(
        "What were sales?",
        [SimpleNamespace(page_number=2, content="Sales: 120")],
        [{"role": "user", "content": "Earlier question"}],
    )
    assert isinstance(result, llm.LLMResponse)
    assert result.answer == "OK" and result.model == MODELS[1]
    assert result.total_tokens == result.prompt_tokens == result.completion_tokens == 0
    body = json.loads(captured[0].content)
    assert body["messages"][1]["content"] == "Earlier question"
    assert "[Page 2]\nSales: 120" in body["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_stream_retains_text_iterator_and_logs_actual_model(
    llm, monkeypatch, caplog
):
    captured = []
    monkeypatch.setattr(
        llm,
        "async_client",
        lambda: async_client(
            key="synthetic-not-a-real-key",
            transport=httpx.MockTransport(lambda req: response(req, captured)),
        ),
    )
    with caplog.at_level("INFO", logger="backend.openrouter_client"):
        chunks = [chunk async for chunk in llm.stream_answer("Question", [])]
    assert chunks == ["OK"]
    assert MODELS[1] in caplog.text

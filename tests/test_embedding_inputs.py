"""Embedding preprocessing contracts without downloading a model."""

from unittest.mock import Mock

import numpy as np
import pytest

from backend import ingest, retrieval


def test_e5_prefixes_and_normalization(monkeypatch):
    monkeypatch.setattr(ingest.settings, "embedding_model", "intfloat/multilingual-e5-small")
    model = Mock()
    model.encode.return_value = np.zeros((1, 384))
    monkeypatch.setattr(ingest, "get_embedding_model", lambda: model)
    ingest.embed_chunks([ingest.TextChunk("Arabic or English", 1, 0)])
    model.encode.assert_called_once_with(
        ["passage: Arabic or English"], show_progress_bar=False, normalize_embeddings=True
    )
    model.encode.reset_mock()
    model.encode.return_value = np.zeros(384)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: model)
    retrieval.embed_query("question")
    model.encode.assert_called_once_with(
        "query: question", show_progress_bar=False, normalize_embeddings=True
    )


def test_non_e5_model_is_unprefixed(monkeypatch):
    monkeypatch.setattr(ingest.settings, "embedding_model", "all-MiniLM-L6-v2")
    assert ingest.embedding_input("text") == "text"
    assert ingest.embedding_input("question", is_query=True) == "question"


def test_model_dimension_mismatch_fails_before_storage(monkeypatch):
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 768
    monkeypatch.setattr(ingest, "_embedding_model", None)
    monkeypatch.setattr(ingest, "SentenceTransformer", lambda _: model)
    with pytest.raises(ValueError, match="dimension"):
        ingest.get_embedding_model()
    assert ingest._embedding_model is None

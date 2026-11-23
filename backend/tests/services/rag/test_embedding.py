"""Dense Ollama embeddings validate batch shape and never pretend to be sparse."""

import json

import httpx
import pytest

from app.core.config import settings
from app.services.rag import embedding


def test_ollama_batches_texts_and_preserves_dense_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict] = []
    client_type = httpx.Client

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"embeddings": [[1.0] + [0.0] * 1023 for _ in body["input"]]})

    monkeypatch.setattr(embedding.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(settings, "RAG_OLLAMA_EMBED_MODEL", "bge-m3")
    vectors = embedding.OllamaEmbedder().encode(["first", "second", "third"], batch_size=2)
    assert len(vectors) == 3 and all(len(vector) == 1024 for vector in vectors)
    assert [len(item["input"]) for item in requests] == [2, 1]
    assert all(item["model"] == "bge-m3" and item["truncate"] is False for item in requests)


@pytest.mark.parametrize("vectors", [[], [[1.0, 0.0]], [[float("nan")] * 1024]])
def test_ollama_rejects_invalid_vectors(monkeypatch: pytest.MonkeyPatch, vectors: list) -> None:
    client_type = httpx.Client
    monkeypatch.setattr(
        embedding.httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=json.dumps({"embeddings": vectors}), headers={"content-type": "application/json"}
                )
            )
        ),
    )
    with pytest.raises(ValueError):
        embedding.OllamaEmbedder().encode(["test"])


def test_ollama_cannot_supply_hybrid_sparse_weights() -> None:
    with pytest.raises(ValueError, match="FlagEmbedding"):
        embedding.OllamaEmbedder().encode_with_sparse(["test"])

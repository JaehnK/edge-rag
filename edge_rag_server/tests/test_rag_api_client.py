from __future__ import annotations

import pytest

from src.Adapters.RagApiClient import RagApiClient


def test_default_endpoint_matches_cpp_rag_server():
    assert (
        RagApiClient(base_url="http://localhost:8000")._build_url().endswith("/query")
    )


def test_build_payload_includes_contract_fields_and_advanced():
    payload = RagApiClient._build_payload(
        question="hello",
        dataset="scifact",
        split="test",
        sample_id="q1",
        advanced="hyde",
    )

    assert payload == {
        "question": "hello",
        "dataset": "scifact",
        "split": "test",
        "sample_id": "q1",
        "advanced": "hyde",
    }


def test_parse_rag_query_contract_response():
    answer = RagApiClient._parse_answer(
        {
            "status": "ok",
            "answer": "response",
            "retrieved_contexts": [
                {
                    "rank": 1,
                    "doc_id": "d1",
                    "chunk_id": 42,
                    "score": 0.91,
                    "text": "context",
                }
            ],
        }
    )

    assert answer.answer == "response"
    assert answer.contexts[0].doc_id == "d1"
    assert answer.contexts[0].text == "context"
    assert answer.contexts[0].score == 0.91


def test_parse_legacy_context_response():
    answer = RagApiClient._parse_answer(
        {
            "answer": "response",
            "contexts": [{"id": "d1", "score": 0.91, "text": "context"}],
        }
    )

    assert answer.contexts[0].doc_id == "d1"


def test_parse_error_response_raises_message():
    with pytest.raises(RuntimeError, match="blank"):
        RagApiClient._parse_answer(
            {
                "status": "error",
                "error": {"message": "Request field question must not be blank."},
            }
        )


def test_ask_sets_client_latency_and_reuses_client(monkeypatch):
    client = RagApiClient(base_url="http://localhost:8000")

    class _FakeResponse:
        is_error = False

        def json(self):
            return {"status": "ok", "answer": "response", "retrieved_contexts": []}

    class _FakeHttpxClient:
        instances = 0

        def __init__(self, *args, **kwargs):
            type(self).instances += 1
            self.is_closed = False

        def post(self, url, json):
            return _FakeResponse()

        def close(self):
            self.is_closed = True

    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeHttpxClient)

    first = client.ask("q1")
    second = client.ask("q2")

    assert first.client_latency_ms is not None
    assert first.client_latency_ms >= 0.0
    assert second.client_latency_ms is not None
    # keep-alive: both requests must share one underlying httpx.Client
    assert _FakeHttpxClient.instances == 1
    client.close()
    assert client._client.is_closed


def test_answer_vs_reference_ragas_metrics_are_rejected():
    from src.EvalCLI import _build_metrics

    with pytest.raises(ValueError, match="passage-level gold documents"):
        _build_metrics(["answer_correctness"])


def test_error_status_surfaces_server_message():
    import httpx

    response = httpx.Response(
        500,
        json={
            "status": "error",
            "error": {"message": "Request field question must not be blank."},
        },
        request=httpx.Request("POST", "http://localhost:8000/query"),
    )

    with pytest.raises(RuntimeError, match="blank"):
        RagApiClient._raise_for_status(response)


def test_error_status_without_body_falls_back_to_httpx_error():
    import httpx

    response = httpx.Response(
        502,
        text="bad gateway",
        request=httpx.Request("POST", "http://localhost:8000/query"),
    )

    with pytest.raises(httpx.HTTPStatusError):
        RagApiClient._raise_for_status(response)

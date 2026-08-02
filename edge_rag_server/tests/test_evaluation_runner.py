from __future__ import annotations

from src.Adapters import BeirEvaluator, RagasEvaluator
from src.Application import EvaluationRunner
from src.DomainModels import BEIRSample, EvalSummary


class _FakeLoader:
    def load(self, dataset_name: str, split: str = "test"):
        sample = BEIRSample(query_id="q1", query="hello", gold_doc_ids=["d1"])
        corpus = {"d1": {"title": "Gold", "text": "reference context"}}
        return [sample], corpus, {}, {"q1": {"d1": 1}}


class _FakeApiClient:
    def __init__(self):
        self.last_kwargs = None

    def ask(self, question: str, **kwargs):
        self.last_kwargs = kwargs

        class _Ctx:
            def __init__(self):
                self.doc_id = "d1"
                self.text = "context"
                self.score = 0.9

        class _Answer:
            def __init__(self):
                self.answer = "response"
                self.contexts = [_Ctx()]
                self.raw = {
                    "trace": {
                        "latency_ms": {"total": 12},
                        "memory_mb": 100.0,
                    },
                    "run_config": {"retrieval": {"top_k": 3}},
                }

        return _Answer()


def test_evaluation_runner_sync_flow():
    """Run the sync flow and assert summary basics."""
    api_client = _FakeApiClient()
    runner = EvaluationRunner(
        loader=_FakeLoader(),
        api_client=api_client,
        beir_evaluator=BeirEvaluator(),
        ragas_evaluator=RagasEvaluator(),
    )

    summary = runner.execute(dataset_name="dummy", use_async=False, advanced="hyde")
    assert isinstance(summary, EvalSummary)
    assert summary.record_count == 1
    assert summary.success_count == 1
    assert summary.error_count == 0
    assert summary.advanced == "hyde"
    assert "precision@1" in summary.beir_metrics
    assert summary.performance_metrics["latency_ms.total"]["p95"] == 12.0
    assert summary.run_config["retrieval"]["top_k"] == 3
    assert summary.records[0].reference == "Gold\nreference context"
    assert api_client.last_kwargs["dataset"] == "dummy"
    assert api_client.last_kwargs["split"] == "test"
    assert api_client.last_kwargs["sample_id"] == "q1"
    assert api_client.last_kwargs["advanced"] == "hyde"


def test_summary_records_request_shape_and_client_latency():
    """use_async/concurrency land in the summary; client wall-time is aggregated."""
    api_client = _FakeApiClient()
    runner = EvaluationRunner(
        loader=_FakeLoader(),
        api_client=api_client,
        beir_evaluator=BeirEvaluator(),
        ragas_evaluator=RagasEvaluator(),
    )

    summary = runner.execute(dataset_name="dummy", use_async=False, concurrency=8)
    # Sync path is sequential regardless of --concurrency: record effective 1.
    assert summary.use_async is False
    assert summary.concurrency == 1
    # _FakeApiClient's answer has no client_latency_ms -> no aggregate key.
    assert "client_latency_ms" not in summary.performance_metrics


def test_client_latency_is_aggregated_when_present():
    class _TimedApiClient(_FakeApiClient):
        def ask(self, question: str, **kwargs):
            answer = super().ask(question, **kwargs)
            answer.client_latency_ms = 345.0
            return answer

    runner = EvaluationRunner(
        loader=_FakeLoader(),
        api_client=_TimedApiClient(),
        beir_evaluator=BeirEvaluator(),
        ragas_evaluator=RagasEvaluator(),
    )

    summary = runner.execute(dataset_name="dummy")
    assert summary.records[0].client_latency_ms == 345.0
    assert summary.performance_metrics["client_latency_ms"]["mean"] == 345.0


def test_technique_effect_counts_noops_separately_from_fallbacks():
    """no-op과 fallback은 결과가 같아 집계로만 구분된다."""
    from src.Application.EvaluationRunner import _summarize_technique_effect
    from src.DomainModels import EvalRecord

    def record(effect: str) -> EvalRecord:
        return EvalRecord(
            query_id="q",
            user_input="q",
            response="a",
            trace={"technique_effect": effect},
        )

    summary = _summarize_technique_effect(
        [record("applied"), record("noop"), record("noop"), record("fallback")]
    )

    assert summary["applied"] == 1
    assert summary["noop"] == 2
    assert summary["fallback"] == 1
    assert summary["attempted"] == 4
    assert summary["noop_rate"] == 0.5


def test_technique_effect_rates_ignore_the_baseline_mode():
    """advanced=none은 돌릴 transform이 없으므로 비율의 분모에서 빠진다."""
    from src.Application.EvaluationRunner import _summarize_technique_effect
    from src.DomainModels import EvalRecord

    baseline = [
        EvalRecord(
            query_id="q",
            user_input="q",
            response="a",
            trace={"technique_effect": "not_applicable"},
        )
        for _ in range(3)
    ]

    summary = _summarize_technique_effect(baseline)

    assert summary["not_applicable"] == 3
    assert "noop_rate" not in summary, "baseline must not be reported as 100% no-op"

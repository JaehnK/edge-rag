from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from src.Adapters import BeirEvaluator, BeirLoader, RagApiClient, RagasEvaluator
from src.DomainModels import EvalRecord, EvalSummary

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRunner:
    """
    평가 파이프라인을 구성하는 오케스트레이터.
    Orchestrator that wires the evaluation pipeline.
    """

    loader: BeirLoader
    api_client: RagApiClient
    beir_evaluator: BeirEvaluator
    ragas_evaluator: RagasEvaluator

    async def _build_records_async(
        self,
        samples: list,
        corpus: dict,
        concurrency: int,
        dataset_name: str,
        split: str,
        advanced: str | None,
    ) -> list[EvalRecord]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(sample) -> EvalRecord:
            try:
                async with semaphore:
                    answer = await self.api_client.ask_async(
                        sample.query,
                        dataset=dataset_name,
                        split=split,
                        sample_id=sample.query_id,
                        advanced=advanced,
                    )
                return _record_from_answer(sample, corpus, answer, advanced)
            except Exception as exc:
                logger.exception(
                    "action=ask_sample status=error sample_id=%s", sample.query_id
                )
                return _error_record(sample, corpus, exc)

        tasks = [_one(sample) for sample in samples]
        return await asyncio.gather(*tasks)

    def _build_records_sync(
        self,
        samples: list,
        corpus: dict,
        dataset_name: str,
        split: str,
        advanced: str | None,
    ) -> list[EvalRecord]:
        records: list[EvalRecord] = []
        for sample in samples:
            try:
                answer = self.api_client.ask(
                    sample.query,
                    dataset=dataset_name,
                    split=split,
                    sample_id=sample.query_id,
                    advanced=advanced,
                )
                records.append(_record_from_answer(sample, corpus, answer, advanced))
            except Exception as exc:
                logger.exception(
                    "action=ask_sample status=error sample_id=%s", sample.query_id
                )
                records.append(_error_record(sample, corpus, exc))
        return records

    def execute(
        self,
        dataset_name: str,
        split: str = "test",
        ragas_metrics: list[Any] | None = None,
        use_async: bool = False,
        concurrency: int = 8,
        advanced: str | None = None,
        limit: int | None = None,
    ) -> EvalSummary:
        """
        BEIR + RAGAS 평가를 실행하고 요약 결과를 반환한다.
        Run BEIR + RAGAS evaluation and return a summary.
        """
        start = time.perf_counter()
        logger.info(
            "action=run_evaluation status=start dataset=%s split=%s "
            "use_async=%s concurrency=%s",
            dataset_name,
            split,
            use_async,
            concurrency,
        )

        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if use_async and concurrency > 1:
            # Concurrent requests contend for the single edge device's CPU, so
            # trace latency includes queueing — fine for throughput smoke runs,
            # invalid for the paper's latency tables (keep concurrency=1 there).
            logger.warning(
                "action=run_evaluation status=latency_contention_warning "
                "concurrency=%s — trace latency will include queueing/contention; "
                "do NOT use this run for latency reporting",
                concurrency,
            )
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")

        samples, corpus, _queries, _qrels = self.loader.load(
            dataset_name=dataset_name, split=split
        )
        if limit is not None:
            samples = samples[:limit]

        if use_async:
            records = asyncio.run(
                self._build_records_async(
                    samples,
                    corpus,
                    concurrency=concurrency,
                    dataset_name=dataset_name,
                    split=split,
                    advanced=advanced,
                )
            )
        else:
            records = self._build_records_sync(
                samples,
                corpus,
                dataset_name=dataset_name,
                split=split,
                advanced=advanced,
            )

        logger.info(
            "action=build_records status=ok records=%s",
            len(records),
        )

        successful_records = [record for record in records if record.error is None]
        # Compute BEIR over successful records only, so that BEIR and RAGAS in the
        # same summary row are means over the SAME query subset. Error records have
        # empty retrieval and otherwise scored 0 on every BEIR metric, silently
        # deflating nDCG/recall while RAGAS was already computed over successful
        # records only.
        beir_metrics = self.beir_evaluator.evaluate(successful_records)
        ragas_metrics_result: dict[str, float] = {}
        ragas_judge: dict = {}
        if ragas_metrics and successful_records:
            ragas_metrics_result = self.ragas_evaluator.evaluate(
                records=successful_records, metrics=ragas_metrics
            )
            judge_config = getattr(self.ragas_evaluator, "judge_config", None)
            ragas_judge = judge_config() if callable(judge_config) else {}
            ragas_judge["metrics"] = [
                str(getattr(m, "name", type(m).__name__)) for m in ragas_metrics
            ]

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "action=run_evaluation status=ok records=%s elapsed_ms=%s",
            len(records),
            elapsed_ms,
        )
        return EvalSummary(
            dataset=dataset_name,
            split=split,
            advanced=advanced or "none",
            use_async=use_async,
            concurrency=concurrency if use_async else 1,
            beir_metrics=beir_metrics,
            ragas_metrics=ragas_metrics_result,
            ragas_judge=ragas_judge,
            performance_metrics=_summarize_performance(successful_records),
            technique_effect=_summarize_technique_effect(successful_records),
            run_config=_first_run_config(successful_records),
            record_count=len(records),
            success_count=len(successful_records),
            error_count=len(records) - len(successful_records),
            elapsed_ms=elapsed_ms,
            records=records,
        )


def _record_from_answer(
    sample, corpus: dict, answer, advanced: str | None
) -> EvalRecord:
    reference_contexts = _reference_contexts(sample.gold_doc_ids, corpus)
    raw_value = getattr(answer, "raw", {})
    raw = raw_value if isinstance(raw_value, dict) else {}
    reported_advanced = raw.get("advanced")
    if reported_advanced is not None and reported_advanced != (advanced or "none"):
        raise RuntimeError(
            f"server advanced mismatch: requested {advanced or 'none'}, "
            f"got {reported_advanced}"
        )
    return EvalRecord(
        query_id=sample.query_id,
        user_input=sample.query,
        response=answer.answer,
        reference="\n\n".join(reference_contexts),
        reference_context_ids=sample.gold_doc_ids,
        reference_contexts=reference_contexts,
        retrieved_context_ids=[ctx.doc_id for ctx in answer.contexts],
        retrieved_contexts=[ctx.text for ctx in answer.contexts],
        retrieved_context_scores=[
            getattr(ctx, "score", 0.0) for ctx in answer.contexts
        ],
        retrieval_doc_ids=[
            ctx.doc_id for ctx in getattr(answer, "pre_transform_contexts", []) or []
        ],
        trace=raw.get("trace") if isinstance(raw.get("trace"), dict) else {},
        run_config=(
            raw.get("run_config") if isinstance(raw.get("run_config"), dict) else {}
        ),
        client_latency_ms=getattr(answer, "client_latency_ms", None),
    )


def _error_record(sample, corpus: dict, exc: Exception) -> EvalRecord:
    reference_contexts = _reference_contexts(sample.gold_doc_ids, corpus)
    return EvalRecord(
        query_id=sample.query_id,
        user_input=sample.query,
        response="",
        reference="\n\n".join(reference_contexts),
        reference_context_ids=sample.gold_doc_ids,
        reference_contexts=reference_contexts,
        error=f"{type(exc).__name__}: {exc}",
    )


def _reference_contexts(doc_ids: list[str], corpus: dict) -> list[str]:
    contexts: list[str] = []
    for doc_id in doc_ids:
        document = corpus.get(doc_id) or {}
        title = str(document.get("title") or "").strip()
        text = str(document.get("text") or "").strip()
        context = "\n".join(part for part in (title, text) if part)
        contexts.append(context)
    return contexts


def _first_run_config(records: list[EvalRecord]) -> dict:
    return next((record.run_config for record in records if record.run_config), {})


def _summarize_performance(records: list[EvalRecord]) -> dict:
    values: dict[str, list[float]] = {}
    for record in records:
        _append_number(values, "client_latency_ms", record.client_latency_ms)
        trace = record.trace
        latency = trace.get("latency_ms") if isinstance(trace, dict) else None
        if isinstance(latency, dict):
            for name, value in latency.items():
                _append_number(values, f"latency_ms.{name}", value)
        for name in (
            "memory_mb",
            "output_tokens_per_sec",
            "prompt_tokens_before_truncation",
            "prompt_tokens",
            "output_tokens",
            # Auxiliary-generation tokens (query_rewrite / hyde /
            # contextual_compression). Without these the token and throughput
            # columns understate exactly the modes that spend an extra LLM call.
            "aux_prompt_tokens",
            "aux_output_tokens",
            "total_prompt_tokens",
            "total_output_tokens",
        ):
            _append_number(
                values, name, trace.get(name) if isinstance(trace, dict) else None
            )
    return {name: _statistics(numbers) for name, numbers in sorted(values.items())}


def _summarize_technique_effect(records: list[EvalRecord]) -> dict:
    """
    기법이 실제로 실행됐는지 집계한다.
    Count how often the mode's transform actually changed anything.

    A no-op produces the same result as advanced=none while being averaged into
    the technique's score, so a mode can look ineffective when it simply did not
    run. SciFact 2026-08-01: query_rewrite returned a byte-identical query on
    252 of 300 claims and reported 300 successes.
    """
    counts = {"applied": 0, "noop": 0, "fallback": 0, "not_applicable": 0}
    for record in records:
        trace = record.trace
        effect = trace.get("technique_effect") if isinstance(trace, dict) else None
        if effect in counts:
            counts[effect] += 1
    # not_applicable is advanced=none, where there is no transform to run. Rating
    # it against the total would report the baseline as 100% no-op.
    attempted = counts["applied"] + counts["noop"] + counts["fallback"]
    if attempted:
        counts["attempted"] = attempted
        counts["applied_rate"] = counts["applied"] / attempted
        counts["noop_rate"] = counts["noop"] / attempted
    return counts


def _append_number(values: dict[str, list[float]], name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    number = float(value)
    if math.isfinite(number):
        values.setdefault(name, []).append(number)


def _statistics(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.DomainModels import EvalRecord

logger = logging.getLogger(__name__)


@dataclass
class RagasEvaluator:
    """
    RAGAS를 이용해 응답 품질 지표를 계산하는 평가기.
    Evaluator that computes response quality metrics using RAGAS.
    """

    # Judge configuration. Left as None, RAGAS falls back to its own default
    # model, which means the paper could not say which judge produced the
    # numbers — and a default change between versions would silently move them.
    # Pin it explicitly and echo it into the summary.
    llm_model: str | None = None
    embedding_model: str | None = None
    # RAGAS asks the judge for temperature 0.01 to keep scoring deterministic,
    # but some models accept only their default. gpt-5.6-luna rejects anything
    # but 1 with:
    #   400 Unsupported value: 'temperature' does not support 0.01 with this
    #   model. Only the default (1) value is supported.
    # and every metric silently became NaN. Set this to the value the judge
    # accepts; None keeps the RAGAS default.
    judge_temperature: float | None = None
    # Judge sampling seed. At temperature 1 the judge is no longer deterministic
    # by construction, so pin a seed to keep repeat runs comparable.
    judge_seed: int | None = None
    # RAGAS's per-job timeout counts queuing time, not just the API call. With
    # the default 180s and 16 workers, context_precision jobs were killed before
    # they ran (9 of 10 rows NaN) even though each API call takes ~2s. Raise the
    # timeout, and/or lower max_workers so jobs are not starved.
    judge_timeout: int | None = None
    judge_max_workers: int | None = None
    # answer_relevancy embeds the generated question and the original one, so it
    # needs an embedding model on top of the judge LLM. The other three metrics
    # (faithfulness, context_precision, context_recall) are LLM-only.
    raise_exceptions: bool = False
    # Filled in by evaluate(); judge tokens actually consumed by the last run.
    last_token_usage: dict = field(default_factory=dict)

    def judge_config(self) -> dict:
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "raise_exceptions": self.raise_exceptions,
            "temperature": self.judge_temperature,
            "seed": self.judge_seed,
            "timeout": self.judge_timeout,
            "max_workers": self.judge_max_workers,
            "token_usage": dict(self.last_token_usage),
        }

    def _build_judge(self):
        """
        지정된 judge LLM/임베딩을 RAGAS가 받는 형태로 만든다.
        Build the pinned judge LLM/embeddings in the form RAGAS accepts.
        """
        llm = None
        embeddings = None
        if self.llm_model:
            from langchain_openai import ChatOpenAI

            # Import from ragas.llms.base, not ragas.llms: the latter re-exports
            # a DeprecationHelper shim that forwards calls but cannot be
            # subclassed ("__init__() takes 3 positional arguments but 4 given").
            from ragas.llms.base import LangchainLLMWrapper

            kwargs: dict = {"model": self.llm_model}
            if self.judge_seed is not None:
                kwargs["seed"] = self.judge_seed
            chat = ChatOpenAI(**kwargs)

            if self.judge_temperature is None:
                llm = LangchainLLMWrapper(chat)
            else:
                # RAGAS decides the temperature per call via get_temperature(),
                # which overrides anything set on the client. Pinning it here is
                # the only place the model-accepted value survives.
                temperature = self.judge_temperature

                class _FixedTemperatureWrapper(LangchainLLMWrapper):
                    def get_temperature(self, n: int) -> float:
                        return temperature

                llm = _FixedTemperatureWrapper(chat)
        if self.embedding_model:
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings.base import LangchainEmbeddingsWrapper

            embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model=self.embedding_model)
            )
        return llm, embeddings

    def build_dataset(self, records: list[EvalRecord]):
        """
        EvalRecord 리스트를 RAGAS EvaluationDataset으로 변환한다.
        Convert EvalRecord list into a RAGAS EvaluationDataset.
        """
        try:
            from ragas import EvaluationDataset, SingleTurnSample
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "ragas is required to build evaluation datasets"
            ) from exc

        samples = []
        for record in records:
            samples.append(
                SingleTurnSample(
                    user_input=record.user_input,
                    response=record.response,
                    reference=record.reference,
                    retrieved_contexts=record.retrieved_contexts,
                    reference_contexts=record.reference_contexts,
                    retrieved_context_ids=record.retrieved_context_ids,
                    reference_context_ids=record.reference_context_ids,
                )
            )
        return EvaluationDataset(samples=samples)

    def evaluate(
        self,
        records: list[EvalRecord],
        metrics: list[Any],
    ) -> dict[str, float]:
        """
        RAGAS metrics를 실행하고 요약 점수를 반환한다.
        Run RAGAS metrics and return aggregate scores.
        """
        if not metrics:
            raise ValueError("metrics must be provided for RAGAS evaluation")
        start = time.perf_counter()

        try:
            from ragas import evaluate as ragas_evaluate
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ragas is required to evaluate metrics") from exc

        dataset = self.build_dataset(records)
        llm, embeddings = self._build_judge()
        # Count what the judge actually consumed instead of estimating it. The
        # judge is billed per token and this run makes tens of thousands of
        # calls, so the number belongs in the report next to the scores.
        token_usage_parser = None
        if llm is not None:
            try:
                from ragas.cost import get_token_usage_for_openai

                token_usage_parser = get_token_usage_for_openai
            except ImportError:  # pragma: no cover - optional in older ragas
                token_usage_parser = None
        logger.info(
            "action=eval_ragas status=judge llm=%s embeddings=%s raise_exceptions=%s",
            self.llm_model or "(ragas default)",
            self.embedding_model or "(ragas default)",
            self.raise_exceptions,
        )
        run_config = None
        if self.judge_timeout is not None or self.judge_max_workers is not None:
            from ragas.run_config import RunConfig

            defaults = RunConfig()
            run_config = RunConfig(
                timeout=self.judge_timeout or defaults.timeout,
                max_workers=self.judge_max_workers or defaults.max_workers,
                seed=self.judge_seed if self.judge_seed is not None else defaults.seed,
            )

        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=self.raise_exceptions,
            token_usage_parser=token_usage_parser,
            run_config=run_config,
        )
        self.last_token_usage = _extract_token_usage(result)
        df = result.to_pandas()
        score_rows = _metric_rows(df, metrics)
        if len(score_rows) != len(records):
            message = (
                f"RAGAS row count mismatch: expected {len(records)}, "
                f"got {len(score_rows)}"
            )
            raise RuntimeError(message)
        for record, scores in zip(records, score_rows, strict=True):
            record.ragas_scores = scores
        summary = _summarize_metrics(df, metrics)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "action=eval_ragas status=ok records=%s metrics=%s elapsed_ms=%s",
            len(records),
            len(metrics),
            elapsed_ms,
        )
        return summary


def _extract_token_usage(result: Any) -> dict:
    """
    RAGAS 결과에서 judge 토큰 사용량을 꺼낸다(집계가 없으면 빈 dict).
    Pull judge token usage off the RAGAS result; empty when not tracked.
    """
    try:
        usage = result.total_tokens()
    except Exception:  # pragma: no cover - parser absent or unsupported llm
        return {}
    # total_tokens() returns a TokenUsage, or a list of them for multi-LLM runs.
    if isinstance(usage, list):
        input_tokens = sum(getattr(u, "input_tokens", 0) or 0 for u in usage)
        output_tokens = sum(getattr(u, "output_tokens", 0) or 0 for u in usage)
    else:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
    }


def _summarize_metrics(df: pd.DataFrame, metrics: list[Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    total = int(len(df))
    for metric in metrics:
        name = str(getattr(metric, "name", metric.__class__.__name__))
        if name not in df.columns:
            logger.warning("action=eval_ragas status=metric_missing metric=%s", name)
            continue
        values = pd.to_numeric(df[name], errors="coerce")
        n_valid = int(values.notna().sum())
        # Record how many rows actually contributed. RAGAS judge failures/parse
        # errors become NaN and are skipped by .mean(); without these counts a mean
        # over the easy 40% of rows is indistinguishable from a full-coverage mean.
        summary[f"{name}__n_valid"] = float(n_valid)
        summary[f"{name}__n_total"] = float(total)
        if n_valid < total:
            logger.warning(
                "action=eval_ragas status=partial metric=%s nan_rows=%s total=%s",
                name,
                total - n_valid,
                total,
            )
        if n_valid > 0:
            summary[name] = float(values.mean())
    return summary


def _metric_rows(df: pd.DataFrame, metrics: list[Any]) -> list[dict[str, float]]:
    names = [
        str(getattr(metric, "name", metric.__class__.__name__)) for metric in metrics
    ]
    rows: list[dict[str, float]] = []
    for _, source in df.iterrows():
        row: dict[str, float] = {}
        for name in names:
            value = pd.to_numeric(pd.Series([source.get(name)]), errors="coerce").iloc[
                0
            ]
            if pd.notna(value):
                row[name] = float(value)
        rows.append(row)
    return rows

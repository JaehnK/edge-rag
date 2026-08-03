from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.Adapters import (
    BeirEvaluator,
    BeirLoader,
    RagApiClient,
    RagasEvaluator,
    ReportWriter,
)
from src.Application import EvaluationRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file() -> Path | None:
    """
    프로젝트 루트부터 위로 올라가며 첫 .env를 로드한다.
    Load the nearest .env, searching from the project root upwards.

    The three repos of this project sit side by side under one workspace, so a
    shared credential naturally lives at the workspace root rather than inside
    any one repo. Looking only at the project root left such a key unread and
    the run died with "api_key must be set" despite the file existing.
    Real environment variables still win over anything in the file.
    """
    for directory in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate
    return None


# In this pipeline `reference` is the concatenation of the qrels-positive gold
# DOCUMENTS (passage-level), not a gold short answer. Metrics that grade the
# generated answer against `reference` as if it were a gold answer would compare
# a one-line answer with thousands of words of corpus text and score
# systematically near zero — looking like a model defect. Only context-grounded
# metrics (faithfulness, context_precision/recall, ...) are valid here (P1-8).
_ANSWER_VS_REFERENCE_METRICS = {
    "answercorrectness",
    "answer_correctness",
    "factualcorrectness",
    "factual_correctness",
    "answersimilarity",
    "answer_similarity",
    "semanticsimilarity",
    "semantic_similarity",
    "answeraccuracy",
    "answer_accuracy",
    "rougescore",
    "rouge_score",
    "bleuscore",
    "bleu_score",
    "exactmatch",
    "exact_match",
    "stringpresence",
    "string_presence",
    "nonllmstringsimilarity",
    "non_llm_string_similarity",
}


def _build_metrics(metric_names: list[str]) -> list[Any]:
    if not metric_names:
        return []
    try:
        from ragas import metrics as ragas_metrics_module
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ragas is required to use ragas metrics") from exc

    metrics: list[Any] = []
    for name in metric_names:
        if name.lower() in _ANSWER_VS_REFERENCE_METRICS:
            raise ValueError(
                f"ragas metric '{name}' compares the answer against `reference`, "
                "but in this pipeline `reference` is passage-level gold documents "
                "(qrels), not a gold answer — scores would be meaningless. "
                "Use context-grounded metrics instead "
                "(e.g. faithfulness, context_precision, context_recall)."
            )
        if not hasattr(ragas_metrics_module, name):
            raise ValueError(f"Unknown ragas metric: {name}")
        metric_obj = getattr(ragas_metrics_module, name)
        metric = metric_obj() if isinstance(metric_obj, type) else metric_obj
        metrics.append(metric)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEIR + RAGAS evaluation CLI")
    parser.add_argument("--dataset", required=True, help="BEIR dataset name")
    parser.add_argument("--split", default="test", help="BEIR split (default: test)")
    parser.add_argument(
        "--data-dir",
        default="beir_data",
        help="Directory containing BEIR datasets (default: beir_data)",
    )
    parser.add_argument("--base-url", required=True, help="FastAPI base URL")
    parser.add_argument("--endpoint", default="/query", help="RAG API endpoint")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--advanced",
        default=None,
        help="Advanced RAG method name to send to the server",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async HTTP requests",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent requests when async is enabled",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N qrels-backed queries (smoke runs only)",
    )
    parser.add_argument(
        "--ragas-metric",
        dest="ragas_metrics",
        action="append",
        default=[],
        help="RAGAS metric name from ragas.metrics (repeatable)",
    )
    parser.add_argument(
        "--ragas-llm-model",
        help="Judge LLM for RAGAS metrics (e.g. gpt-4o-mini). "
        "Omit to use the RAGAS default, which is not reproducible.",
    )
    parser.add_argument(
        "--ragas-embedding-model",
        help="Embedding model for RAGAS metrics that need one (answer_relevancy)",
    )
    parser.add_argument(
        "--ragas-judge-temperature",
        type=float,
        help="Temperature for the judge LLM. RAGAS defaults to 0.01; some models "
        "(e.g. gpt-5.6-luna) accept only 1 and return every score as NaN otherwise.",
    )
    parser.add_argument(
        "--ragas-judge-seed",
        type=int,
        help="Sampling seed for the judge LLM, for repeatable scores when the "
        "model forces a non-zero temperature",
    )
    parser.add_argument(
        "--ragas-judge-timeout",
        type=int,
        help="Per-job timeout for RAGAS (seconds). Counts queuing, not just the "
        "API call; the 180s default silently NaN'd context_precision.",
    )
    parser.add_argument(
        "--ragas-judge-max-workers",
        type=int,
        help="Concurrent judge jobs (RAGAS default 16)",
    )
    parser.add_argument(
        "--ragas-raise-exceptions",
        action="store_true",
        help="Abort the run if any judge call fails, instead of scoring it NaN "
        "and averaging over the survivors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the complete run, including per-query records, to JSON",
    )
    return parser.parse_args()


def run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # The RAGAS judge reads OPENAI_API_KEY from the environment, so the .env has
    # to actually be loaded — otherwise it is an inert file and the run dies with
    # "api_key client option must be set" despite the key being right there.
    env_path = _load_env_file()
    if env_path:
        logging.getLogger(__name__).info("action=load_env status=ok path=%s", env_path)
    args = _parse_args()

    loader = BeirLoader(data_dir=args.data_dir)
    api_client = RagApiClient(
        base_url=args.base_url,
        endpoint=args.endpoint,
        timeout_sec=args.timeout,
    )
    beir_evaluator = BeirEvaluator()
    ragas_evaluator = RagasEvaluator(
        llm_model=args.ragas_llm_model,
        embedding_model=args.ragas_embedding_model,
        judge_temperature=args.ragas_judge_temperature,
        judge_seed=args.ragas_judge_seed,
        judge_timeout=args.ragas_judge_timeout,
        judge_max_workers=args.ragas_judge_max_workers,
        raise_exceptions=args.ragas_raise_exceptions,
    )
    if args.ragas_metrics and not args.ragas_llm_model:
        logging.getLogger(__name__).warning(
            "action=run status=unpinned_judge — RAGAS metrics requested without "
            "--ragas-llm-model; scores will come from the RAGAS default judge and "
            "are not reproducible across versions"
        )

    runner = EvaluationRunner(
        loader=loader,
        api_client=api_client,
        beir_evaluator=beir_evaluator,
        ragas_evaluator=ragas_evaluator,
    )

    metrics = _build_metrics(args.ragas_metrics)
    try:
        summary = runner.execute(
            dataset_name=args.dataset,
            split=args.split,
            ragas_metrics=metrics,
            use_async=args.use_async,
            concurrency=args.concurrency,
            advanced=args.advanced,
            limit=args.limit,
        )
    finally:
        api_client.close()

    if args.output:
        writer = ReportWriter(output_dir=str(args.output.parent))
        writer.write_json(summary, filename=args.output.name)

    payload = asdict(summary)
    payload.pop("records", None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if summary.error_count else 0

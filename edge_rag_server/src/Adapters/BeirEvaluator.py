from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from src.DomainModels import EvalRecord

logger = logging.getLogger(__name__)


@dataclass
class BeirEvaluator:
    """
    검색 결과를 기반으로 간단한 IR 지표를 계산하는 평가기.
    Evaluator that computes simple IR metrics from retrieval results.
    """

    def evaluate(
        self,
        records: list[EvalRecord],
        k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> dict[str, float]:
        """
        EvalRecord 리스트로부터 precision/recall@k를 계산한다.
        Compute document-level IR metrics from EvalRecord list.
        """
        start = time.perf_counter()
        metrics: dict[str, float] = {}
        # No records to evaluate over (e.g. every request errored): return an empty
        # dict rather than fabricated 0.0 metrics, which read as a valid-but-terrible
        # system. Callers should surface error_count separately.
        if not records:
            logger.warning("action=eval_beir status=empty records=0")
            return metrics

        for k in k_values:
            precision_sum = 0.0
            recall_sum = 0.0
            ndcg_sum = 0.0
            map_sum = 0.0
            mrr_sum = 0.0

            for record in records:
                # Score the retrieval, not the post-transform prompt contexts:
                # for contextual_compression the latter is a single synthetic
                # "compressed" chunk that matches no qrel, which would report 0
                # for a mode whose retrieval was in fact identical to `none`.
                provenance = record.retrieval_doc_ids or record.retrieved_context_ids
                retrieved = _unique(provenance)[:k]
                gold = set(record.reference_context_ids)
                relevant_ranks = [
                    rank
                    for rank, doc_id in enumerate(retrieved, start=1)
                    if doc_id in gold
                ]
                hit_count = len(relevant_ranks)

                precision = hit_count / k
                recall = hit_count / len(gold) if gold else 0.0
                # NOTE: binary-relevance nDCG (gain 1.0 per hit). Graded qrels are
                # collapsed to a relevant/not-relevant set upstream (BeirLoader),
                # so on graded datasets (e.g. trec-covid, nfcorpus) these values do
                # NOT match official BEIR/pytrec_eval nDCG. For paper-grade graded
                # scores, delegate to pytrec_eval / beir EvaluateRetrieval. See
                # CODE_REVIEW_FIXES.md P1-6.
                dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
                ideal_hits = min(len(gold), k)
                idcg = sum(
                    1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)
                )
                ndcg = dcg / idcg if idcg else 0.0
                # Average precision divides by R = |gold| (trec_eval/BEIR
                # convention), NOT min(|gold|, k) which inflated MAP for queries
                # with more than k relevant docs.
                average_precision = (
                    sum(
                        hits_so_far / rank
                        for hits_so_far, rank in enumerate(relevant_ranks, start=1)
                    )
                    / len(gold)
                    if gold
                    else 0.0
                )
                reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

                precision_sum += precision
                recall_sum += recall
                ndcg_sum += ndcg
                map_sum += average_precision
                mrr_sum += reciprocal_rank

            metrics[f"precision@{k}"] = precision_sum / len(records)
            metrics[f"recall@{k}"] = recall_sum / len(records)
            metrics[f"ndcg@{k}"] = ndcg_sum / len(records)
            metrics[f"map@{k}"] = map_sum / len(records)
            metrics[f"mrr@{k}"] = mrr_sum / len(records)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "action=eval_beir status=ok records=%s k_values=%s elapsed_ms=%s",
            len(records),
            k_values,
            elapsed_ms,
        )
        return metrics


def _unique(doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))

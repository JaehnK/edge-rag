from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beir.datasets.data_loader import GenericDataLoader

from src.DomainModels import BEIRSample

logger = logging.getLogger(__name__)


@dataclass
class BeirLoader:
    """
    BEIR 데이터셋을 로드하고 질의 단위 샘플로 변환하는 어댑터.
    Adapter that loads a BEIR dataset and converts it into query-level samples.
    """

    data_dir: str = "beir_data"

    def load(
        self,
        dataset_name: str,
        split: str = "test",
    ) -> tuple[
        list[BEIRSample],
        dict[str, Any],
        dict[str, str],
        dict[str, dict[str, int]],
    ]:
        """
        BEIR 데이터를 로드하고 질의 단위 샘플을 생성한다.
        Load BEIR data and build query-level samples.
        """
        start = time.perf_counter()
        data_folder = Path(self.data_dir) / dataset_name
        corpus_file = "corpus.jsonl"
        separate_corpus = Path(self.data_dir) / f"{dataset_name}_corpus" / corpus_file
        if not (data_folder / corpus_file).exists() and separate_corpus.exists():
            corpus_file = str(separate_corpus.resolve())
        loader = GenericDataLoader(
            data_folder=str(data_folder), corpus_file=corpus_file
        )
        corpus, queries, qrels = loader.load(split=split)

        samples: list[BEIRSample] = []
        for query_id, query in queries.items():
            gold_doc_ids = [
                doc_id
                for doc_id, score in (qrels.get(query_id) or {}).items()
                if score > 0
            ]

            samples.append(
                BEIRSample(
                    query_id=str(query_id),
                    query=str(query),
                    gold_doc_ids=[str(doc_id) for doc_id in gold_doc_ids],
                )
            )

        if not samples:
            raise ValueError(
                f"BEIR load produced 0 samples for dataset={dataset_name} "
                f"split={split}; check data_dir/qrels path and split name"
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "action=load_beir status=ok dataset=%s split=%s samples=%s elapsed_ms=%s",
            dataset_name,
            split,
            len(samples),
            elapsed_ms,
        )
        return samples, corpus, queries, qrels

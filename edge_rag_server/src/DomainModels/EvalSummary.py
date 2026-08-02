from dataclasses import dataclass, field

from .EvalRecord import EvalRecord


@dataclass
class EvalSummary:
    """
    전체 평가 결과를 요약한 메트릭 묶음 객체.
    Metrics bundle summarizing the full evaluation run.
    """

    dataset: str = ""
    split: str = ""
    advanced: str = "none"
    # Request-shape of the run that produced the latency numbers. trace latency
    # from concurrent requests includes queueing/CPU contention on the single
    # edge device, so rows with concurrency > 1 must never be mixed into a
    # sequential latency table — record the setting so they stay separable.
    use_async: bool = False
    concurrency: int = 1
    beir_metrics: dict = field(default_factory=dict)
    ragas_metrics: dict = field(default_factory=dict)
    # Which judge produced ragas_metrics. Without this the quality numbers are
    # not reproducible — a different judge model gives different scores for the
    # same answers, and RAGAS's own default can change between versions.
    ragas_judge: dict = field(default_factory=dict)
    performance_metrics: dict = field(default_factory=dict)
    # Whether the mode's transform actually ran, per query. A transform that
    # hands back its input is averaged into the technique's score while having
    # done the work of advanced=none, so a flat result can mean "did not run"
    # rather than "did not help".
    technique_effect: dict = field(default_factory=dict)
    run_config: dict = field(default_factory=dict)
    record_count: int = 0
    success_count: int = 0
    error_count: int = 0
    elapsed_ms: int = 0
    records: list[EvalRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        요약 메트릭의 기본 유효성을 점검한다.
        Validate basic invariants for the summary metrics.
        """
        if self.record_count < 0:
            raise ValueError("record_count must be >= 0")
        if self.success_count < 0 or self.error_count < 0:
            raise ValueError("success_count and error_count must be >= 0")
        if self.success_count + self.error_count != self.record_count:
            raise ValueError("success_count + error_count must equal record_count")

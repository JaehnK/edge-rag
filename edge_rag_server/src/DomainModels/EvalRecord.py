from dataclasses import dataclass, field


@dataclass
class EvalRecord:
    """
    단일 질의에 대한 입력, 검색, 응답 정보를 묶은 평가 단위 객체.
    Evaluation unit aggregating input, retrieval, and response for one query.
    """

    query_id: str
    user_input: str
    response: str
    reference: str = ""
    reference_context_ids: list[str] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)
    retrieved_context_ids: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_context_scores: list[float] = field(default_factory=list)
    # Doc ids of what RETRIEVAL actually returned, which differs from
    # retrieved_context_ids when a mode rewrites the contexts before generation:
    # contextual_compression merges them into one synthetic chunk whose id is
    # "compressed". IR metrics must score the retrieval, so BEIR reads this;
    # empty means retrieved_context_ids already is the retrieval output.
    retrieval_doc_ids: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    run_config: dict = field(default_factory=dict)
    # End-to-end HTTP wall time measured by the client (includes server-side
    # process startup / model load that trace.latency_ms.total omits).
    client_latency_ms: float | None = None
    ragas_scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        """
        평가 레코드의 기본 일관성을 점검한다.
        Validate basic consistency of the evaluation record.
        """
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if len(self.retrieved_context_ids) != len(self.retrieved_contexts):
            raise ValueError(
                "retrieved_context_ids and retrieved_contexts must be same length"
            )
        if self.reference_contexts and len(self.reference_context_ids) != len(
            self.reference_contexts
        ):
            raise ValueError(
                "reference_context_ids and reference_contexts must be same length"
            )
        if self.retrieved_context_scores and len(self.retrieved_context_ids) != len(
            self.retrieved_context_scores
        ):
            raise ValueError(
                "retrieved_context_ids and retrieved_context_scores must be same length"
            )

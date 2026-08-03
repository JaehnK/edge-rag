from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from src.DomainModels import ApiAnswer, RetrievedContext

logger = logging.getLogger(__name__)


def _truncate(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


@dataclass
class RagApiClient:
    """
    FastAPI 엔드포인트 호출을 담당하는 어댑터.
    Adapter responsible for calling the FastAPI endpoint.
    """

    base_url: str
    endpoint: str = "/query"
    timeout_sec: float = 30.0

    def _get_client(self) -> httpx.Client:
        """
        동기 httpx 클라이언트를 재사용한다(keep-alive).
        Reuse a sync httpx client across requests (keep-alive).
        """
        client = getattr(self, "_client", None)
        if client is None or client.is_closed:
            client = httpx.Client(timeout=self.timeout_sec)
            self._client = client
        return client

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None and not client.is_closed:
            client.close()

    def _build_url(self) -> str:
        """
        base_url과 endpoint를 결합해 호출 URL을 구성한다.
        Build the full request URL from base_url and endpoint.
        """
        endpoint = (
            self.endpoint if self.endpoint.startswith("/") else f"/{self.endpoint}"
        )
        return f"{self.base_url.rstrip('/')}{endpoint}"

    @staticmethod
    def _build_payload(
        question: str,
        dataset: str | None = None,
        split: str | None = None,
        sample_id: str | None = None,
        advanced: str | None = None,
    ) -> dict:
        payload = {
            "question": question,
            "dataset": dataset,
            "split": split,
            "sample_id": sample_id,
        }
        if advanced:
            payload["advanced"] = advanced
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """
        비정상 상태코드일 때 서버 에러 메시지를 살려서 예외를 던진다.
        Raise with the server's own error message on a non-2xx status.
        """
        if not response.is_error:
            return
        # The server returns 5xx with rag_query's structured
        # {"status":"error","error":{"message":...}} body for child failures.
        # Surfacing that message (instead of a bare HTTPStatusError) keeps the
        # per-record error strings diagnosable.
        try:
            payload = response.json()
            message = (payload.get("error") or {}).get("message")
        except Exception:
            message = None
        if message:
            raise RuntimeError(f"HTTP {response.status_code}: {message}")
        response.raise_for_status()

    @staticmethod
    def _parse_answer(payload: dict) -> ApiAnswer:
        """
        API 응답 JSON을 ApiAnswer로 변환한다.
        Convert the API JSON payload into ApiAnswer.
        """
        if payload.get("status") == "error":
            error = payload.get("error") or {}
            message = error.get("message") or "RAG API returned error status"
            raise RuntimeError(str(message))

        def _to_contexts(rows) -> list[RetrievedContext]:
            out: list[RetrievedContext] = []
            for ctx in rows or []:
                out.append(
                    RetrievedContext(
                        doc_id=str(ctx.get("doc_id") or ctx.get("id", "")),
                        text=str(ctx.get("text") or ""),
                        # `or 0.0` (not a default arg) so an explicit
                        # "score": null in the payload — which compressed
                        # contexts now carry — doesn't raise TypeError and turn
                        # the whole query into an error record.
                        score=float(ctx.get("score") or 0.0),
                    )
                )
            return out

        raw_contexts = payload.get("retrieved_contexts")
        if raw_contexts is None:
            raw_contexts = payload.get("contexts", [])

        return ApiAnswer(
            answer=str(payload.get("answer") or ""),
            contexts=_to_contexts(raw_contexts),
            pre_transform_contexts=_to_contexts(payload.get("pre_transform_contexts")),
            raw=payload,
        )

    def ask(
        self,
        question: str,
        dataset: str | None = None,
        split: str | None = None,
        sample_id: str | None = None,
        advanced: str | None = None,
    ) -> ApiAnswer:
        """
        동기 방식으로 질문을 보내고 응답을 반환한다.
        Send a question synchronously and return the response.
        """
        payload = self._build_payload(
            question=question,
            dataset=dataset,
            split=split,
            sample_id=sample_id,
            advanced=advanced,
        )
        start = time.perf_counter()
        client = self._get_client()
        try:
            response = client.post(self._build_url(), json=payload)
            self._raise_for_status(response)
            answer = self._parse_answer(response.json())
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            answer.client_latency_ms = float(elapsed_ms)
            logger.info(
                "action=ask status=ok elapsed_ms=%s query=%s",
                elapsed_ms,
                _truncate(question),
            )
            return answer
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "action=ask status=error elapsed_ms=%s query=%s",
                elapsed_ms,
                _truncate(question),
            )
            raise

    async def ask_async(
        self,
        question: str,
        dataset: str | None = None,
        split: str | None = None,
        sample_id: str | None = None,
        advanced: str | None = None,
    ) -> ApiAnswer:
        """
        비동기 방식으로 질문을 보내고 응답을 반환한다.
        Send a question asynchronously and return the response.
        """
        payload = self._build_payload(
            question=question,
            dataset=dataset,
            split=split,
            sample_id=sample_id,
            advanced=advanced,
        )
        start = time.perf_counter()
        # Async clients are created per call on purpose: each execute() spins up
        # its own event loop via asyncio.run, and an AsyncClient outliving its
        # loop breaks on reuse. The sequential (sync) path is the one used for
        # latency reporting and does reuse a client.
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            try:
                response = await client.post(self._build_url(), json=payload)
                self._raise_for_status(response)
                answer = self._parse_answer(response.json())
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                answer.client_latency_ms = float(elapsed_ms)
                logger.info(
                    "action=ask_async status=ok elapsed_ms=%s query=%s",
                    elapsed_ms,
                    _truncate(question),
                )
                return answer
            except Exception:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                logger.exception(
                    "action=ask_async status=error elapsed_ms=%s query=%s",
                    elapsed_ms,
                    _truncate(question),
                )
                raise

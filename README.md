# edge-rag

Raspberry Pi 온디바이스 RAG 실험 저장소.

- `edge_rag_server/` — Python 평가 하네스 (구 `JaehnK/RAG_evaluator`, 히스토리 포함)
- `edge-rag-experiments/` — C++ RAG 파이프라인. submodule로 `JaehnK/edge-rag-experiments`(`jjo2889/edge-rag-experiments`의 포크)를 가리킨다

```
git clone --recurse-submodules https://github.com/JaehnK/edge-rag.git
cd edge-rag/edge-rag-experiments
git remote add upstream https://github.com/jjo2889/edge-rag-experiments.git
```

실험 코드는 포크에서 작업하고, 원 저장소에는 PR로 올린다.

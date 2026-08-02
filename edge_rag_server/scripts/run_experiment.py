from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODES = (
    "none",
    "query_rewrite",
    "hyde",
    "rerank",
    "sentence_window",
    "contextual_compression",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same BEIR evaluation across Advanced RAG modes."
    )
    parser.add_argument("--dataset", required=True, help="BEIR dataset name")
    parser.add_argument("--split", default="test", help="BEIR split")
    parser.add_argument("--data-dir", default="beir_data", help="BEIR data directory")
    parser.add_argument("--base-url", required=True, help="RAG API base URL")
    parser.add_argument("--endpoint", default="/query", help="RAG API endpoint")
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=MODES,
        help="Mode to run; repeat to select modes (default: all)",
    )
    parser.add_argument("--async", dest="use_async", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, help="Smoke-run query limit")
    parser.add_argument(
        "--ragas-metric",
        dest="ragas_metrics",
        action="append",
        default=[],
    )
    parser.add_argument("--ragas-llm-model", help="Judge LLM for RAGAS metrics")
    parser.add_argument(
        "--ragas-embedding-model",
        help="Embedding model for RAGAS metrics that need one (answer_relevancy)",
    )
    parser.add_argument("--ragas-judge-temperature", type=float)
    parser.add_argument("--ragas-judge-seed", type=int)
    parser.add_argument("--ragas-judge-timeout", type=int)
    parser.add_argument("--ragas-judge-max-workers", type=int)
    parser.add_argument("--ragas-raise-exceptions", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip modes whose result JSON already exists, so a multi-hour run "
        "can be resumed after an interruption without redoing finished modes",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    modes = args.modes or list(MODES)
    output_dir = _output_dir(args.output_dir, args.dataset)
    commands = [
        _evaluation_command(args, mode, output_dir / f"{mode}.json") for mode in modes
    ]
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "started_at": _now(),
        "completed_at": None,
        "evaluator_commit": _git_commit(),
        "dataset": args.dataset,
        "split": args.split,
        "data_dir": args.data_dir,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "modes": modes,
        "use_async": args.use_async,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "limit": args.limit,
        "ragas_metrics": args.ragas_metrics,
        "ragas_llm_model": args.ragas_llm_model,
        "ragas_embedding_model": args.ragas_embedding_model,
        "ragas_judge_temperature": args.ragas_judge_temperature,
        "ragas_judge_seed": args.ragas_judge_seed,
        "runs": [],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    failed = False
    for mode, command in zip(modes, commands, strict=True):
        result_path = output_dir / f"{mode}.json"
        if args.skip_existing and result_path.exists():
            print(f"{mode}: skipped (result already exists)", flush=True)
            manifest["runs"].append(
                {
                    "mode": mode,
                    "status": "skipped",
                    "returncode": 0,
                    "elapsed_ms": 0,
                    "output": f"{mode}.json",
                    "stdout_log": f"{mode}.stdout.log",
                    "stderr_log": f"{mode}.stderr.log",
                    "command": command,
                }
            )
            _write_json(manifest_path, manifest)
            _write_summary_csv(output_dir, manifest["runs"])
            continue

        started = time.perf_counter()
        print(f"{mode}: started at {_now()}", flush=True)
        # Stream the child's output straight to disk instead of buffering it in
        # memory until the mode finishes. A mode here runs for hours; with
        # capture_output the log file stayed empty that whole time, so there was
        # no way to see progress or notice a stall.
        stdout_path = output_dir / f"{mode}.stdout.log"
        stderr_path = output_dir / f"{mode}.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                stdout=out,
                stderr=err,
                check=False,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        run = {
            "mode": mode,
            "status": "ok" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "elapsed_ms": elapsed_ms,
            "output": f"{mode}.json",
            "stdout_log": f"{mode}.stdout.log",
            "stderr_log": f"{mode}.stderr.log",
            "command": command,
        }
        manifest["runs"].append(run)
        failed = failed or result.returncode != 0
        _write_json(manifest_path, manifest)
        _write_summary_csv(output_dir, manifest["runs"])
        print(
            f"{mode}: {run['status']} ({elapsed_ms/3600000:.2f} h) at {_now()}",
            flush=True,
        )

    manifest["status"] = "error" if failed else "ok"
    manifest["completed_at"] = _now()
    _write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}")
    return 1 if failed else 0


def _evaluation_command(
    args: argparse.Namespace, mode: str, output_path: Path
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--data-dir",
        args.data_dir,
        "--base-url",
        args.base_url,
        "--endpoint",
        args.endpoint,
        "--advanced",
        mode,
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
        "--output",
        str(output_path),
    ]
    if args.use_async:
        command.append("--async")
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    for metric in args.ragas_metrics:
        command.extend(("--ragas-metric", metric))
    if args.ragas_llm_model:
        command.extend(("--ragas-llm-model", args.ragas_llm_model))
    if args.ragas_embedding_model:
        command.extend(("--ragas-embedding-model", args.ragas_embedding_model))
    if args.ragas_judge_temperature is not None:
        command.extend(("--ragas-judge-temperature", str(args.ragas_judge_temperature)))
    if args.ragas_judge_seed is not None:
        command.extend(("--ragas-judge-seed", str(args.ragas_judge_seed)))
    if args.ragas_judge_timeout is not None:
        command.extend(("--ragas-judge-timeout", str(args.ragas_judge_timeout)))
    if args.ragas_judge_max_workers is not None:
        command.extend(("--ragas-judge-max-workers", str(args.ragas_judge_max_workers)))
    if args.ragas_raise_exceptions:
        command.append("--ragas-raise-exceptions")
    return command


def _default_output_dir(dataset: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "reports" / f"{dataset}-{timestamp}"


def _output_dir(path: Path | None, dataset: str) -> Path:
    if path is None:
        return _default_output_dir(dataset)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _write_summary_csv(output_dir: Path, runs: list[dict]) -> None:
    rows: list[dict] = []
    for run in runs:
        row = {
            "mode": run["mode"],
            "status": run["status"],
            "returncode": run["returncode"],
            "elapsed_ms": run["elapsed_ms"],
        }
        result_path = output_dir / run["output"]
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            # Carry the request shape into the table itself: latency columns are
            # only comparable across rows that ran with the same concurrency, and
            # a CSV pasted into the paper loses the manifest that recorded it.
            for name in (
                "record_count",
                "success_count",
                "error_count",
                "use_async",
                "concurrency",
            ):
                row[name] = result.get(name)
            _flatten(row, "beir", result.get("beir_metrics") or {})
            _flatten(row, "ragas", result.get("ragas_metrics") or {})
            _flatten(row, "performance", result.get("performance_metrics") or {})
            _flatten(row, "technique", result.get("technique_effect") or {})
        rows.append(row)

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    summary_path = output_dir / "summary.csv"
    temporary_path = summary_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(summary_path)


def _flatten(target: dict, prefix: str, value: dict) -> None:
    for key, child in value.items():
        name = f"{prefix}.{key}"
        if isinstance(child, dict):
            _flatten(target, name, child)
        else:
            target[name] = child


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

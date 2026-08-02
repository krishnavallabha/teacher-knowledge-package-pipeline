"""Run the full pipeline against a batch of documents and summarize results.

This harness is intentionally file-driven so you can point it at a folder of
real documents (for example, downloaded NCERT chapters you already have
rights to use) and compare structural/grounding results across multiple
inputs in one run.

Usage:
    python scripts/evaluate_pipeline.py --input-dir samples
    python scripts/evaluate_pipeline.py --files doc1.pdf doc2.docx doc3.txt

The script prints a compact JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".ppt", ".txt", ".md"}


@dataclass
class EvaluationResult:
    file: str
    job_id: str | None
    subject: str | None
    grade: str | None
    structural_passed: bool | None
    grounding_passed: bool | None
    validation_issue_count: int
    grounding_issue_count: int
    error: str | None = None


async def _run_one(path: Path, curriculum_board: str | None = None) -> EvaluationResult:
    from app.models.schemas import DocumentTypeHint, UserContext
    from app.orchestrator import run_pipeline

    file_bytes = path.read_bytes()
    user_context = None
    if curriculum_board:
        user_context = UserContext(
            document_type_hint=DocumentTypeHint.NOT_SURE,
            curriculum_board=curriculum_board,
        )
    try:
        final_tkp = None
        async for stage, progress, tkp in run_pipeline(path.name, file_bytes, user_context):
            final_tkp = tkp
        assert final_tkp is not None
        return EvaluationResult(
            file=str(path),
            job_id=final_tkp.job_id,
            subject=final_tkp.metadata.subject if final_tkp.metadata else None,
            grade=final_tkp.metadata.grade if final_tkp.metadata else None,
            structural_passed=final_tkp.validation.passed if final_tkp.validation else None,
            grounding_passed=final_tkp.grounding.passed if final_tkp.grounding else None,
            validation_issue_count=len(final_tkp.validation.issues) if final_tkp.validation else 0,
            grounding_issue_count=len(final_tkp.grounding.ungrounded_claims) if final_tkp.grounding else 0,
        )
    except Exception as exc:  # noqa: BLE001 - harness should continue across failures
        return EvaluationResult(
            file=str(path),
            job_id=None,
            subject=None,
            grade=None,
            structural_passed=None,
            grounding_passed=None,
            validation_issue_count=0,
            grounding_issue_count=0,
            error=str(exc),
        )


async def _run_batch(paths: Iterable[Path], curriculum_board: str | None = None):
    results = []
    for path in paths:
        results.append(await _run_one(path, curriculum_board=curriculum_board))
    return results


def _collect_files(input_dir: Path | None, files: list[Path]) -> list[Path]:
    if files:
        return [p for p in files if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.exists()]
    if input_dir is None:
        return []
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate the teacher knowledge pipeline.")
    parser.add_argument("--input-dir", type=Path, help="Directory of documents to evaluate")
    parser.add_argument("--files", nargs="*", type=Path, help="Explicit document paths")
    parser.add_argument("--curriculum-board", type=str, default=None, help="Optional board hint to pass through")
    parser.add_argument(
        "--provider", type=str, default=None, choices=["groq", "gemini", "anthropic", "openai"],
        help="Force a specific LLM provider instead of auto-detecting from which API key is set",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Optional model override for whichever provider is active "
             "(sets GROQ_MODEL, GEMINI_MODEL, ANTHROPIC_MODEL, or OPENAI_MODEL "
             "depending on --provider or auto-detection)",
    )
    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        provider = args.provider or os.environ.get("LLM_PROVIDER", "groq")
        env_var = {
            "groq": "GROQ_MODEL", "gemini": "GEMINI_MODEL",
            "anthropic": "ANTHROPIC_MODEL", "openai": "OPENAI_MODEL",
        }[provider]
        os.environ[env_var] = args.model

    paths = _collect_files(args.input_dir, args.files or [])
    if not paths:
        print(json.dumps({"error": "No supported input files found."}, indent=2))
        return 1

    results = asyncio.run(_run_batch(paths, curriculum_board=args.curriculum_board))
    summary = {
        "total": len(results),
        "passed_structural": sum(1 for r in results if r.structural_passed is True),
        "passed_grounding": sum(1 for r in results if r.grounding_passed is True),
        "errors": sum(1 for r in results if r.error),
        "results": [asdict(r) for r in results],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

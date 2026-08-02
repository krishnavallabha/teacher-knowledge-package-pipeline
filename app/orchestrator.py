"""
Pipeline orchestrator.

This is the one place that knows the full stage order, mirroring the
suggested architecture: API Gateway -> Upload -> Document Intelligence ->
Classification -> Knowledge Extraction -> Teaching Planner -> Content/
Activity/Assessment Generators -> Validation -> Grounding Validation ->
Publishing. Each stage function is a pure transformation on the
TeacherKnowledgePackage; the orchestrator's only job is sequencing them and
reporting progress, which is what makes it possible to add a stage or
reorder one without touching any other stage's code.

user_context carries the answers to the small set of clarifying questions
asked before the pipeline starts (Stage 0, handled entirely on the frontend
and in the API route -- there is no separate "stage" for it here since it
produces no new pipeline state, just inputs that bias later stages). Every
field on it is optional, so a teacher who skips the clarifying questions
gets exactly the old behavior of inferring everything from the document.

Implemented as an async generator that yields (stage, progress, tkp)
tuples, which the FastAPI route below turns directly into an SSE stream --
this is the same object driving both progress reporting and the final
result, so there's no risk of the two drifting out of sync.
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator

from app.models.schemas import PipelineStage, TeacherKnowledgePackage, UserContext
from app.services import (
    classification,
    document_intelligence,
    gap_analysis,
    grounding_validation,
    grounding_correction,
    knowledge_extraction,
    period_generation,
    publishing,
    teaching_planner,
    validation,
)

OUTPUT_ROOT = Path("job_outputs")

# job_id -> dict of output file paths, populated by the publishing stage.
# A plain module-level dict is enough for a single-process demo deployment;
# a real production version would put this in the same DB as job status.
JOB_ARTIFACTS: dict[str, dict] = {}


async def run_pipeline(
    filename: str, file_bytes: bytes, user_context: UserContext | None = None
) -> AsyncGenerator[tuple[PipelineStage, int, TeacherKnowledgePackage], None]:
    job_id = str(uuid.uuid4())[:8]
    tkp = TeacherKnowledgePackage(job_id=job_id, source_filename=filename, user_context=user_context)

    # Each LLM-calling stage is synchronous (a blocking SDK call), so we
    # push it to a worker thread with asyncio.to_thread rather than blocking
    # the event loop -- this is what lets the progress stream actually
    # flush to the frontend between stages instead of arriving all at once
    # at the end.

    doc_type_hint = user_context.document_type_hint if user_context else None

    tkp.stage, tkp.progress = PipelineStage.DOCUMENT_INTELLIGENCE, 5
    yield tkp.stage, tkp.progress, tkp

    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue[int] = asyncio.Queue()

    def _report_doc_progress(progress: int) -> None:
        loop.call_soon_threadsafe(progress_queue.put_nowait, progress)

    parse_task = asyncio.create_task(
        asyncio.to_thread(
            document_intelligence.parse_document,
            filename,
            file_bytes,
            doc_type_hint or document_intelligence.DocumentTypeHint.NOT_SURE,
            _report_doc_progress,
        )
    )

    last_progress = tkp.progress
    while not parse_task.done():
        try:
            doc_progress = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
        except TimeoutError:
            continue

        mapped_progress = 5 + max(0, min(10, doc_progress // 10))
        if mapped_progress > last_progress:
            last_progress = mapped_progress
            tkp.progress = mapped_progress
            yield tkp.stage, tkp.progress, tkp

    parsed = await parse_task

    while not progress_queue.empty():
        doc_progress = progress_queue.get_nowait()
        mapped_progress = 5 + max(0, min(10, doc_progress // 10))
        if mapped_progress > last_progress:
            last_progress = mapped_progress
            tkp.progress = mapped_progress
            yield tkp.stage, tkp.progress, tkp

    tkp.progress = 15
    yield tkp.stage, tkp.progress, tkp

    tkp.stage, tkp.progress = PipelineStage.EDUCATIONAL_CLASSIFICATION, 15
    yield tkp.stage, tkp.progress, tkp
    tkp.metadata = await asyncio.to_thread(classification.classify_document, parsed, user_context)

    tkp.stage, tkp.progress = PipelineStage.KNOWLEDGE_EXTRACTION, 30
    yield tkp.stage, tkp.progress, tkp
    tkp.knowledge = await asyncio.to_thread(knowledge_extraction.extract_knowledge, parsed, tkp.metadata)

    tkp.stage, tkp.progress = PipelineStage.TEACHING_PLANNER, 45
    yield tkp.stage, tkp.progress, tkp
    tkp.teaching_plan = await asyncio.to_thread(
        teaching_planner.build_teaching_plan, tkp.knowledge, tkp.metadata, user_context
    )

    tkp.stage, tkp.progress = PipelineStage.PERIOD_GENERATION, 55
    yield tkp.stage, tkp.progress, tkp
    # Default lowered from 2 to 1: Groq's free tier binds on tokens/minute
    # (12000 TPM for llama-3.3-70b-versatile), not requests/minute, and
    # concurrency doesn't help against a token ceiling the way it helps
    # against a request-count ceiling -- two calls in flight at once just
    # means their token budgets land in the same window instead of spread
    # across it. Two concurrent period-generation calls at ~4600 max output
    # tokens each, plus input tokens for both, can alone approach the full
    # 12000 TPM budget before any other stage's calls land in that minute.
    # Raise this via env var if you're on a paid tier with a materially
    # higher TPM limit.
    period_concurrency = max(1, int(os.environ.get("PERIOD_GENERATION_CONCURRENCY", "1")))
    period_semaphore = asyncio.Semaphore(period_concurrency)

    async def _generate_period_artifacts(period_index: int, period):
        async with period_semaphore:
            bundle = await asyncio.to_thread(
                period_generation.generate_period_bundle,
                period,
                tkp.knowledge,
                tkp.metadata,
                parsed,
            )
            return period_index, bundle

    generation_tasks = [
        asyncio.create_task(_generate_period_artifacts(i, period))
        for i, period in enumerate(tkp.teaching_plan.periods)
    ]

    completed = 0
    period_results: list[object | None] = [None] * len(generation_tasks)
    for future in asyncio.as_completed(generation_tasks):
        period_index, bundle = await future
        period_results[period_index] = bundle
        completed += 1
        tkp.progress = 55 + int(25 * completed / len(generation_tasks))
        yield tkp.stage, tkp.progress, tkp

    for bundle in period_results:
        if bundle is None:
            continue
        tkp.period_contents.append(bundle.content)
        tkp.activities.append(bundle.activity)
        tkp.assessments.append(bundle.assessment)

    tkp.stage, tkp.progress = PipelineStage.GAP_ANALYSIS, 88
    yield tkp.stage, tkp.progress, tkp
    tkp.learning_gaps = await asyncio.to_thread(gap_analysis.analyze_gaps, tkp.knowledge, tkp.metadata)

    tkp.stage, tkp.progress = PipelineStage.VALIDATION, 91
    yield tkp.stage, tkp.progress, tkp
    tkp.validation = validation.validate_package(tkp)

    tkp.stage, tkp.progress = PipelineStage.GROUNDING_VALIDATION, 95
    yield tkp.stage, tkp.progress, tkp
    tkp.grounding = await asyncio.to_thread(
        grounding_validation.validate_grounding, tkp, parsed
    )

    if tkp.grounding and not tkp.grounding.passed:
        tkp.progress = 96
        yield tkp.stage, tkp.progress, tkp
        tkp = await asyncio.to_thread(grounding_correction.correct_ungrounded_content, tkp, parsed)
        tkp.grounding = await asyncio.to_thread(
            grounding_validation.validate_grounding, tkp, parsed
        )

    tkp.stage, tkp.progress = PipelineStage.PUBLISHING, 97
    yield tkp.stage, tkp.progress, tkp
    output_dir = OUTPUT_ROOT / job_id
    publish_paths = await asyncio.to_thread(publishing.publish_package, tkp, output_dir)
    JOB_ARTIFACTS[job_id] = publish_paths

    tkp.stage, tkp.progress = PipelineStage.DONE, 100
    yield tkp.stage, tkp.progress, tkp

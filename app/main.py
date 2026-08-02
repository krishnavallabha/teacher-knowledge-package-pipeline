"""
API Gateway for the Teacher Knowledge Package pipeline.

POST /api/generate accepts a document upload plus an optional set of
clarifying-question answers (grade, objectives, style, time constraint,
document type hint) as form fields, and streams progress as Server-Sent
Events, one event per pipeline stage, ending with the final
TeacherKnowledgePackage JSON. GET /api/download/{job_id}/{artifact} serves
the published PDFs and JSON once a job completes.

All clarifying-question fields are optional query/form parameters rather
than a required separate endpoint, so a teacher who just wants to upload
and go can skip them entirely and get the same behavior as before they
existed -- the frontend decides whether to show that step at all.

SSE rather than plain polling because Stage 10's spec explicitly asks for
"a mechanism to stream progress updates for long-running AI jobs to the
frontend" -- SSE is the simplest transport that satisfies that over plain
HTTP without needing a websocket server.
"""
print("========== APP STARTED ==========", flush=True)
import json
from pathlib import Path
import traceback
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import DocumentTypeHint, UserContext
from app.orchestrator import JOB_ARTIFACTS, run_pipeline

app = FastAPI(title="Teacher Knowledge Package Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_JOB_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "job_outputs"
_ARTIFACT_FILENAMES = {
    "json": "TeacherKnowledgePackage.json",
    "lesson_plan_pdf": "LessonPlan.pdf",
    "teacher_guide_pdf": "TeacherGuide.pdf",
    "assessment_book_pdf": "AssessmentBook.pdf",
}





@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    grade_override: str | None = Form(None),
    teaching_objectives: str | None = Form(None),
    teaching_style: str | None = Form(None),
    time_constraint_minutes: int | None = Form(None),
    curriculum_board: str | None = Form(None),
    document_type_hint: str | None = Form(None),
):
    file_bytes = await file.read()
    filename = file.filename or "upload.txt"

    user_context = UserContext(
        grade_override=grade_override or None,
        teaching_objectives=teaching_objectives or None,
        teaching_style=teaching_style or None,
        time_constraint_minutes=time_constraint_minutes,
        curriculum_board=curriculum_board or None,
        document_type_hint=DocumentTypeHint(document_type_hint) if document_type_hint else DocumentTypeHint.NOT_SURE,
    )

    async def event_stream():
        try:
            async for stage, progress, tkp in run_pipeline(filename, file_bytes, user_context):
                payload = {"stage": stage.value, "progress": progress}
                if stage.value == "done":
                    payload["result"] = json.loads(tkp.model_dump_json())
                    payload["artifacts"] = JOB_ARTIFACTS.get(tkp.job_id, {})
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:  # noqa: BLE001 -- surface any pipeline failure to the client
            traceback.print_exc()
            print("PIPELINE FAILED:", repr(e))
            yield f"data: {json.dumps({
        'stage': 'failed',
        'progress': 0,
        'error': str(e)
    })}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/download/{job_id}/{artifact}")
def download(job_id: str, artifact: str):
    paths = JOB_ARTIFACTS.get(job_id, {})
    file_path_str = paths.get(artifact)

    if not file_path_str:
        fallback_name = _ARTIFACT_FILENAMES.get(artifact)
        if fallback_name:
            fallback_path = _JOB_OUTPUTS_DIR / job_id / fallback_name
            if fallback_path.exists():
                file_path_str = str(fallback_path)

    if not file_path_str:
        raise HTTPException(status_code=404, detail="Artifact not found")

    file_path = Path(file_path_str)
    suffix = file_path.suffix.lower()

    media_types = {".pdf": "application/pdf", ".json": "application/json",}
    


    return FileResponse(
    path=file_path,
    filename=file_path.name,
    media_type=media_types.get(suffix, "application/octet-stream"),
)
app.mount(
    "/",
    StaticFiles(directory=_FRONTEND_DIR, html=True),
    name="frontend",
)

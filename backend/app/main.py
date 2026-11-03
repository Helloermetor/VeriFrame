from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    Response,
)
from .services.thumbnails import extract_thumbnail
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4
import os
import shutil
import hashlib
import json
import asyncio
from app.api.model_evaluation import router as model_evaluation_router
from datetime import datetime, timezone
from .services import store
from .services.pipeline import UPLOAD_DIR, run_pipeline, sha256_file
from .services.stability import run_stability
from .services.report import build_analysis_pdf
from .services.evidence_package import build_evidence_package
from .services.verify_evidence import verify_evidence_package

app = FastAPI(title="VeriFrame API", version="0.1.0")
app.include_router(model_evaluation_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("VERIFRAME_CORS_ORIGINS", "http://localhost:5173").split(
        ","
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
# Keep references to recovery tasks while they are running.
_recovery_tasks: set[asyncio.Task] = set()


async def _resume_analysis(analysis_id: str, video_path: str):
    try:
        await asyncio.to_thread(run_pipeline, analysis_id, video_path)
    except Exception as exc:
        store.update(
            analysis_id,
            status="failed",
            stage="failed",
            progress=100,
            error=f"Automatic recovery failed: {exc}",
        )
        store.add_event(
            analysis_id,
            "failed",
            100,
            "Automatic recovery failed",
        )


@app.on_event("startup")
async def recover_interrupted_analyses():
    recovered_count = 0
    missing_count = 0

    for record in store.all_records():
        analysis_id = record.get("analysis_id")
        status = record.get("status")

        if not analysis_id or status not in {"queued", "processing"}:
            continue

        video_path = record.get("video_path")

        if not video_path or not Path(video_path).is_file():
            store.update(
                analysis_id,
                status="failed",
                stage="failed",
                progress=100,
                error="Original uploaded video is unavailable. Automatic recovery was not possible.",
            )
            store.add_event(
                analysis_id,
                "failed",
                100,
                "Recovery failed because the original video is missing",
            )
            missing_count += 1
            continue

        store.update(
            analysis_id,
            status="queued",
            stage="queued",
            progress=0,
            error=None,
        )

        store.add_event(
            analysis_id,
            "queued",
            0,
            "Resuming interrupted analysis after backend restart",
        )

        task = asyncio.create_task(_resume_analysis(analysis_id, video_path))
        _recovery_tasks.add(task)
        task.add_done_callback(_recovery_tasks.discard)

        recovered_count += 1

    print(
        f"VeriFrame recovery: {recovered_count} analysis(es) resumed, "
        f"{missing_count} analysis(es) could not be resumed."
    )


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi"}
MAX_UPLOAD_BYTES = int(os.getenv("VERIFRAME_MAX_UPLOAD_BYTES", str(250 * 1024 * 1024)))


def public_record(record):
    return {k: v for k, v in record.items() if k != "video_path"}


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "veriframe-api", "mode": "integrated"}


@app.get("/api/v1/models")
def models():
    return {
        "models": [
            {
                "id": "gend_clip_l14",
                "role": "Primary detector: GenD CLIP ViT-L/14",
                "status": "configured",
            },
            {
                "id": "xception_custom",
                "role": "Control detector: custom DeepfakeBench Xception",
                "status": "configured",
            },
            {
                "id": "effort",
                "role": "Challenger 1: Effort CLIP ViT-L/14 with SVD adaptation",
                "status": "configured",
            },
            {
                "id": "fakestormer",
                "role": "Challenger 2: FakeSTormer C23 ViT-Base",
                "status": "configured",
            },
        ]
    }


@app.post("/api/v1/analyses", status_code=202)
async def create_analysis(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    consent: bool = Form(False),
):
    if not consent:
        raise HTTPException(400, "Consent is required before analysis.")
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported extension. Allowed: mp4, mov, webm, avi.")

    analysis_id = str(uuid4())
    target = UPLOAD_DIR / f"{analysis_id}{suffix}"
    size = 0
    first = b""
    try:
        with target.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                if not first:
                    first = chunk[:32]
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        413, f"Upload exceeds {MAX_UPLOAD_BYTES} bytes."
                    )
                out.write(chunk)
    finally:
        await video.close()

    # Basic magic-byte checks; ffprobe should still be sandboxed in production.
    valid = (
        (suffix == ".mp4" and b"ftyp" in first[:16])
        or (suffix == ".mov" and b"ftyp" in first[:16])
        or (suffix == ".webm" and first.startswith(b"\x1aE\xdf\xa3"))
        or (suffix == ".avi" and first.startswith(b"RIFF") and b"AVI " in first[:16])
    )
    if not valid:
        target.unlink(missing_ok=True)
        raise HTTPException(
            415, "File signature does not match the selected video type."
        )

    digest = sha256_file(str(target))
    store.create(
        analysis_id,
        {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "filename": Path(video.filename or "upload").name,
            "size_bytes": size,
            "sha256": digest,
            "video_path": str(target),
            "result": None,
            "error": None,
        },
    )
    background_tasks.add_task(run_pipeline, analysis_id, str(target))
    return {"analysis_id": analysis_id, "status": "queued", "sha256": digest}


@app.get("/api/v1/analyses")
def list_analyses():
    records = store.all_records()

    sorted_records = sorted(
        records,
        key=lambda item: item.get("created_at") or "",
        reverse=True,
    )

    return {
        "total": len(sorted_records),
        "analyses": [public_record(record) for record in sorted_records],
    }


@app.get("/api/v1/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    return public_record(record)


@app.get("/api/v1/analyses/{analysis_id}/events")
def events(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    # Snapshot endpoint for clients that cannot use SSE in this starter.
    # Replace with StreamingResponse + event IDs for production SSE.
    return {"events": store.events_after(analysis_id, 0), "status": record["status"]}


@app.get("/api/v1/analyses/{analysis_id}/frames")
def frames(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    return {"frames": (record.get("result") or {}).get("frames", [])}


@app.get("/api/v1/analyses/{analysis_id}/evidence")
def evidence(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    return (record.get("result") or {}).get("evidence", {"status": "pending"})


@app.get("/api/v1/analyses/{analysis_id}/stability")
def stability(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    return (record.get("result") or {}).get("stability", {"status": "pending"})


@app.post("/api/v1/analyses/{analysis_id}/stability")
def run_stability_check(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")

    if record.get("status") != "completed":
        raise HTTPException(409, "Wait until the video analysis has completed.")

    video_path = record.get("video_path")
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(404, "Original uploaded video is unavailable.")

    result = run_stability(video_path)

    current_result = record.get("result") or {}
    current_result["stability"] = result
    store.update(analysis_id, result=current_result)

    return result


@app.get("/api/v1/analyses/{analysis_id}/provenance")
def provenance(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    result = record.get("result") or {}
    return {
        "sha256": record["sha256"],
        "metadata": result.get("metadata"),
        "provenance_evidence": (result.get("evidence") or {}).get(
            "provenance_evidence", []
        ),
        "c2pa_status": "not_checked",
    }


@app.get("/api/v1/analyses/{analysis_id}/report")
def report(analysis_id: str):
    record = store.get(analysis_id)

    if not record:
        raise HTTPException(404, "Analysis not found.")

    if record.get("status") != "completed":
        raise HTTPException(
            409, "Wait until the analysis has completed before generating the PDF."
        )

    try:
        pdf_bytes = build_analysis_pdf(record)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="veriframe_report_{analysis_id[:8]}.pdf"'
                )
            },
        )

    except Exception as exc:
        raise HTTPException(500, f"PDF report generation failed: {str(exc)}")


@app.get("/api/v1/analyses/{analysis_id}/evidence-package")
def download_evidence_package(analysis_id: str):
    record = store.get(analysis_id)

    if not record:
        raise HTTPException(404, "Analysis not found.")

    if record.get("status") != "completed":
        raise HTTPException(
            409,
            "Wait until the analysis has completed before generating the evidence package.",
        )

    try:
        pdf_bytes = build_analysis_pdf(record)
        zip_bytes = build_evidence_package(record, pdf_bytes)

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="veriframe_evidence_{analysis_id[:8]}.zip"'
                )
            },
        )

    except Exception as exc:
        raise HTTPException(500, f"Evidence package generation failed: {str(exc)}")


@app.post("/api/v1/evidence/verify")
async def verify_uploaded_evidence_package(
    package: UploadFile = File(...),
):
    if not package.filename or not package.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a valid ZIP evidence package.")

    try:
        zip_bytes = await package.read()

        if not zip_bytes:
            raise HTTPException(400, "The uploaded ZIP file is empty.")

        if len(zip_bytes) > 100 * 1024 * 1024:
            raise HTTPException(
                413, "Evidence package exceeds the 100 MB upload limit."
            )

        verification = verify_evidence_package(zip_bytes)

        return verification

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Evidence verification failed: {str(exc)}")
    finally:
        await package.close()


@app.get("/api/v1/analyses/{analysis_id}/thumbnail")
def analysis_thumbnail(analysis_id: str, timestamp: float):
    record = store.get(analysis_id)

    if not record:
        raise HTTPException(404, "Analysis not found.")

    if record.get("status") != "completed":
        raise HTTPException(409, "Analysis is not completed.")

    video_path = record.get("video_path")

    if not video_path:
        raise HTTPException(404, "Original video path is unavailable.")

    image_bytes = extract_thumbnail(video_path, timestamp)

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/v1/verify/{report_id}")
def verify_report(report_id: str):
    raise HTTPException(
        501,
        "Report verification becomes available after PDF + manifest generation is implemented.",
    )


@app.delete("/api/v1/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str):
    record = store.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    path = record.get("video_path")
    store.delete(analysis_id)
    if path:
        Path(path).unlink(missing_ok=True)
    return Response(status_code=204)

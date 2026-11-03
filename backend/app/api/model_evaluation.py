from __future__ import annotations

import csv
import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel, Field

from app.services.evaluation_report import (
    build_evaluation_pdf,
    build_comparison_pdf,
)

from app.services.model_evaluation import (
    EVALUATION_ROOT,
    run_model_evaluation,
)

router = APIRouter(
    prefix="/api/v1/model-evaluation",
    tags=["Model Evaluation"],
)

_jobs_lock = threading.Lock()
_active_job_id: str | None = None


class EvaluationRequest(BaseModel):
    split: str = Field(default="test")
    batch_size: int = Field(default=32, ge=1, le=256)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def _job_file(evaluation_id: str) -> Path:
    return EVALUATION_ROOT / evaluation_id / "job.json"


def _save_job(job: dict[str, Any]) -> None:
    """Atomically persist a job record to disk."""
    path = _job_file(job["evaluation_id"])
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(job, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _read_job(evaluation_id: str) -> dict[str, Any] | None:
    path = _job_file(evaluation_id)

    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _all_jobs() -> list[dict[str, Any]]:
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []

    for path in EVALUATION_ROOT.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("evaluation_id"):
                jobs.append(job)
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(
        jobs,
        key=lambda job: job.get("created_at", ""),
        reverse=True,
    )


def _calculate_curves(
    csv_path: Path,
    score_column: str,
    level: str,
) -> dict[str, Any]:
    """
    Calculate ROC and Precision-Recall curves from saved evaluation scores.
    Uses the standard library only.
    """
    labels: list[int] = []
    scores: list[float] = []

    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path.name}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {"true_label", score_column}
        available_columns = set(reader.fieldnames or [])

        if not required_columns.issubset(available_columns):
            raise ValueError(
                f"{csv_path.name} must contain columns: "
                f"{', '.join(sorted(required_columns))}"
            )

        for row in reader:
            try:
                label = int(row["true_label"])
                score = float(row[score_column])
            except (TypeError, ValueError):
                continue

            if label not in (0, 1) or not math.isfinite(score):
                continue

            labels.append(label)
            scores.append(score)

    if not labels:
        raise ValueError(f"No valid evaluation rows found in {csv_path.name}")

    positives = sum(labels)
    negatives = len(labels) - positives

    if positives == 0 or negatives == 0:
        raise ValueError(
            f"{level} curves require both original and manipulated samples."
        )

    # Sort scores from highest to lowest.
    sorted_rows = sorted(
        zip(scores, labels),
        key=lambda item: item[0],
        reverse=True,
    )

    roc_points = [{"fpr": 0.0, "tpr": 0.0, "threshold": None}]
    pr_points = [{"recall": 0.0, "precision": 1.0, "threshold": None}]

    true_positives = 0
    false_positives = 0
    index = 0

    while index < len(sorted_rows):
        threshold = sorted_rows[index][0]

        # Process equal scores together so ties are handled correctly.
        while index < len(sorted_rows) and sorted_rows[index][0] == threshold:
            if sorted_rows[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1

            index += 1

        tpr = true_positives / positives
        fpr = false_positives / negatives
        precision = true_positives / (true_positives + false_positives)

        roc_points.append(
            {
                "fpr": fpr,
                "tpr": tpr,
                "threshold": threshold,
            }
        )

        pr_points.append(
            {
                "recall": tpr,
                "precision": precision,
                "threshold": threshold,
            }
        )

    # Trapezoidal area under the ROC curve.
    roc_auc = sum(
        (roc_points[i]["fpr"] - roc_points[i - 1]["fpr"])
        * (roc_points[i]["tpr"] + roc_points[i - 1]["tpr"])
        / 2
        for i in range(1, len(roc_points))
    )

    # Average Precision: step-wise area under the PR curve.
    average_precision = sum(
        (pr_points[i]["recall"] - pr_points[i - 1]["recall"])
        * pr_points[i]["precision"]
        for i in range(1, len(pr_points))
    )

    return {
        "sample_count": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "roc_curve": roc_points,
        "precision_recall_curve": pr_points,
    }


@router.on_event("startup")
def recover_evaluation_jobs() -> None:
    """
    Recover persisted job history after a backend restart.
    """
    global _active_job_id

    # Import older evaluation outputs that predate persistent job records.
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)

    for summary_path in EVALUATION_ROOT.glob("*/xception_*_summary.json"):
        folder = summary_path.parent
        job_path = folder / "job.json"

        if job_path.exists():
            continue

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            evaluation_id = folder.name
            split = summary.get("split_evaluated", "test")

            legacy_job = {
                "evaluation_id": evaluation_id,
                "status": "completed",
                "created_at": datetime.fromtimestamp(
                    summary_path.stat().st_mtime,
                    timezone.utc,
                ).isoformat(),
                "completed_at": datetime.fromtimestamp(
                    summary_path.stat().st_mtime,
                    timezone.utc,
                ).isoformat(),
                "split": split,
                "batch_size": None,
                "threshold": summary.get("threshold", 0.5),
                "result": {
                    "evaluation_id": evaluation_id,
                    "status": "completed",
                    "model": summary.get("model"),
                    "split": split,
                    "threshold": summary.get("threshold", 0.5),
                    "summary": summary,
                    "artifacts": {
                        "summary_json": str(summary_path),
                        "frame_csv": str(
                            folder / f"xception_{split}_frame_results.csv"
                        ),
                        "video_csv": str(
                            folder / f"xception_{split}_video_results.csv"
                        ),
                    },
                },
                "error": None,
                "migration_note": (
                    "Imported from a completed evaluation created "
                    "before persistent job tracking."
                ),
            }

            _save_job(legacy_job)

        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Could not import legacy evaluation " f"{folder.name}: {exc}")

    interrupted = 0

    with _jobs_lock:
        _active_job_id = None

        for job in _all_jobs():
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                job["error"] = (
                    "Backend restarted before this evaluation was "
                    "confirmed complete. Existing output files "
                    "were preserved."
                )
                _save_job(job)
                interrupted += 1

    print(
        "VeriFrame model evaluation recovery: "
        f"{interrupted} interrupted job(s) marked; "
        f"{len(_all_jobs())} persisted job(s) available."
    )


def _run_job(
    evaluation_id: str,
    request: EvaluationRequest,
) -> None:
    global _active_job_id

    job = _read_job(evaluation_id)
    if job is None:
        return

    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    _save_job(job)

    try:
        result = run_model_evaluation(
            split=request.split,
            batch_size=request.batch_size,
            threshold=request.threshold,
            evaluation_id=evaluation_id,
        )

        job["status"] = "completed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["result"] = result
        job["error"] = None
        _save_job(job)

    except Exception as exc:
        job["status"] = "failed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["error"] = f"{type(exc).__name__}: {exc}"
        _save_job(job)

    finally:
        with _jobs_lock:
            if _active_job_id == evaluation_id:
                _active_job_id = None


@router.post("/run", status_code=202)
def start_model_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
):
    global _active_job_id

    if request.split not in {"train", "val", "validation", "test"}:
        raise HTTPException(
            status_code=400,
            detail="split must be train, val, validation, or test.",
        )

    evaluation_id = str(uuid.uuid4())

    with _jobs_lock:
        if _active_job_id is not None:
            active_job = _read_job(_active_job_id) or {}
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A model evaluation is already running.",
                    "evaluation_id": _active_job_id,
                    "status": active_job.get("status"),
                },
            )

        job = {
            "evaluation_id": evaluation_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": request.split,
            "batch_size": request.batch_size,
            "threshold": request.threshold,
            "result": None,
            "error": None,
        }

        _save_job(job)
        _active_job_id = evaluation_id

    background_tasks.add_task(
        _run_job,
        evaluation_id,
        request,
    )

    return {
        "message": "Model evaluation has been queued.",
        "evaluation_id": evaluation_id,
        "status": "queued",
        "status_url": (f"/api/v1/model-evaluation/{evaluation_id}"),
    }


@router.get("")
def get_latest_model_evaluation():
    jobs = _all_jobs()

    if not jobs:
        return {
            "status": "not_run",
            "message": "No model evaluation has been recorded.",
        }

    return jobs[0]


@router.get("/history")
def get_model_evaluation_history():
    jobs = _all_jobs()

    return {
        "count": len(jobs),
        "evaluations": jobs,
    }


@router.get("/{evaluation_id}/curves")
def get_model_evaluation_curves(evaluation_id: str):
    """
    Return frame-level and video-folder-level ROC and
    Precision-Recall curve data for a completed evaluation.
    """

    # Prevent directory traversal outside the evaluation directory.
    evaluation_folder = (EVALUATION_ROOT / evaluation_id).resolve()
    evaluation_root = EVALUATION_ROOT.resolve()

    try:
        evaluation_folder.relative_to(evaluation_root)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid evaluation ID.",
        )

    job = _read_job(evaluation_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation ID not found.",
        )

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Curves are available only for completed evaluations.",
        )

    result = job.get("result") or {}
    artifacts = result.get("artifacts") or {}

    split = job.get("split") or result.get("split") or "test"

    if split == "validation":
        split = "val"

    def resolve_csv(artifact_key: str, fallback_name: str) -> Path:
        supplied_path = artifacts.get(artifact_key)

        path = (
            Path(supplied_path) if supplied_path else evaluation_folder / fallback_name
        )

        path = path.resolve()

        try:
            path.relative_to(evaluation_folder)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {artifact_key} artifact path.",
            )

        return path

    frame_csv = resolve_csv(
        "frame_csv",
        f"xception_{split}_frame_results.csv",
    )

    video_csv = resolve_csv(
        "video_csv",
        f"xception_{split}_video_results.csv",
    )

    try:
        frame_curves = _calculate_curves(
            csv_path=frame_csv,
            score_column="class1_manipulated_score",
            level="frame-level",
        )

        video_curves = _calculate_curves(
            csv_path=video_csv,
            score_column="mean_class1_manipulated_score",
            level="video-folder-level",
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "evaluation_id": evaluation_id,
        "model": "Custom DeepfakeBench Xception",
        "split": split,
        "score_class": "class 1 - manipulated",
        "frame_level": frame_curves,
        "video_level": video_curves,
        "interpretation_note": (
            "Video-level curves use mean class-1 scores aggregated "
            "per frame folder. These folders are not necessarily "
            "independently verified source videos."
        ),
    }


@router.get("/compare")
def compare_model_evaluations(
    first_id: str,
    second_id: str,
):
    """
    Compare two completed model evaluation runs.
    Metric differences are calculated as second evaluation minus first.
    """

    def validate_id(value: str) -> None:
        try:
            uuid.UUID(value)
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid evaluation ID: {value}",
            )

    validate_id(first_id)
    validate_id(second_id)

    if first_id == second_id:
        raise HTTPException(
            status_code=400,
            detail="Choose two different evaluation runs.",
        )

    first_job = _read_job(first_id)
    second_job = _read_job(second_id)

    if first_job is None or second_job is None:
        raise HTTPException(
            status_code=404,
            detail="One or both evaluation IDs were not found.",
        )

    if (
        first_job.get("status") != "completed"
        or second_job.get("status") != "completed"
    ):
        raise HTTPException(
            status_code=409,
            detail="Both evaluations must be completed before comparison.",
        )

    def extract_summary(job: dict[str, Any]) -> dict[str, Any]:
        result = job.get("result") or {}
        summary = result.get("summary")

        if not isinstance(summary, dict):
            raise HTTPException(
                status_code=422,
                detail=("A completed evaluation is missing its summary metrics."),
            )

        return summary

    first_summary = extract_summary(first_job)
    second_summary = extract_summary(second_job)

    metric_fields = {
        "accuracy": "accuracy",
        "precision": "precision_manipulated_class_1",
        "recall": "recall_manipulated_class_1",
        "f1_score": "f1_manipulated_class_1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    def compare_level(
        first_metrics: dict[str, Any],
        second_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        comparisons = {}

        for display_name, source_key in metric_fields.items():
            first_value = first_metrics.get(source_key)
            second_value = second_metrics.get(source_key)

            valid_values = (
                isinstance(first_value, (int, float))
                and not isinstance(first_value, bool)
                and isinstance(second_value, (int, float))
                and not isinstance(second_value, bool)
            )

            comparisons[display_name] = {
                "first": first_value,
                "second": second_value,
                "delta": (second_value - first_value if valid_values else None),
            }

        return comparisons

    first_frame = first_summary.get("frame_metrics") or {}
    second_frame = second_summary.get("frame_metrics") or {}

    first_video = first_summary.get("video_metrics_mean_score_per_frame_folder") or {}
    second_video = second_summary.get("video_metrics_mean_score_per_frame_folder") or {}

    first_split = first_summary.get(
        "split_evaluated",
        first_job.get("split"),
    )
    second_split = second_summary.get(
        "split_evaluated",
        second_job.get("split"),
    )

    warnings = []

    if first_split != second_split:
        warnings.append(
            "The evaluations use different dataset splits. "
            "Metric differences are not a like-for-like comparison."
        )

    first_checkpoint = first_summary.get("checkpoint_sha256")
    second_checkpoint = second_summary.get("checkpoint_sha256")

    if first_checkpoint == second_checkpoint:
        warnings.append(
            "Both evaluations use the same checkpoint. "
            "Differences may result from evaluation configuration "
            "rather than model-weight changes."
        )
    else:
        warnings.append(
            "The evaluations use different checkpoint hashes. "
            "Verify that the dataset and evaluation protocol are "
            "consistent before attributing metric changes to the model."
        )

    return {
        "first": {
            "evaluation_id": first_id,
            "created_at": first_job.get("created_at"),
            "split": first_split,
            "checkpoint_sha256": first_checkpoint,
        },
        "second": {
            "evaluation_id": second_id,
            "created_at": second_job.get("created_at"),
            "split": second_split,
            "checkpoint_sha256": second_checkpoint,
        },
        "same_split": first_split == second_split,
        "same_checkpoint": first_checkpoint == second_checkpoint,
        "frame_level": compare_level(first_frame, second_frame),
        "video_level": compare_level(first_video, second_video),
        "warnings": warnings,
        "delta_note": "Delta = second evaluation - first evaluation.",
    }


@router.get("/compare/report")
def download_comparison_report(
    first_id: str,
    second_id: str,
):
    """Generate a PDF comparing two completed evaluation runs."""

    comparison = compare_model_evaluations(
        first_id=first_id,
        second_id=second_id,
    )

    try:
        pdf_bytes = build_comparison_pdf(comparison)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    'attachment; filename="veriframe_comparison_report.pdf"'
                )
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison PDF generation failed: {str(exc)}",
        )


@router.get("/{evaluation_id}/report")
def download_evaluation_report(evaluation_id: str):
    """Generate a PDF for one completed model evaluation."""

    job = _read_job(evaluation_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation ID not found.",
        )

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="PDF reports are available only for completed evaluations.",
        )

    try:
        pdf_bytes = build_evaluation_pdf(job)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="veriframe_evaluation_{evaluation_id[:8]}.pdf"'
                )
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation PDF generation failed: {str(exc)}",
        )


@router.get("/{evaluation_id}")
def get_model_evaluation(evaluation_id: str):
    job = _read_job(evaluation_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation ID not found.",
        )

    return job

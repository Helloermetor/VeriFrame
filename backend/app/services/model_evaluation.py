from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Optional external DeepfakeBench evaluation installation.
# Configure VERIFRAME_DEEPFAKEBENCH_ROOT or individual file overrides.
_DEEPFAKEBENCH_VALUE = os.getenv("VERIFRAME_DEEPFAKEBENCH_ROOT")
DEEPFAKEBENCH_ROOT = (
    Path(_DEEPFAKEBENCH_VALUE).expanduser()
    if _DEEPFAKEBENCH_VALUE else None
)

_EVALUATION_PYTHON_DEFAULT = (
    DEEPFAKEBENCH_ROOT
    / ("Scripts" if os.name == "nt" else "bin")
    / ("python.exe" if os.name == "nt" else "python")
    if DEEPFAKEBENCH_ROOT else None
)

def _evaluation_path(variable, default):
    value = os.getenv(variable)
    return Path(value).expanduser() if value else default

MODEL_PYTHON = _evaluation_path(
    "VERIFRAME_XCEPTION_EVALUATION_PYTHON", _EVALUATION_PYTHON_DEFAULT
)
EVALUATION_SCRIPT = _evaluation_path(
    "VERIFRAME_XCEPTION_EVALUATION_SCRIPT",
    DEEPFAKEBENCH_ROOT / "batch_evaluate_xception.py"
    if DEEPFAKEBENCH_ROOT else None,
)
DATASET_CSV = _evaluation_path(
    "VERIFRAME_XCEPTION_EVALUATION_DATASET",
    DEEPFAKEBENCH_ROOT / "datasets" / "rgb" / "FaceForensics++"
    / "simple_frames" / "dataset_source_disjoint.csv"
    if DEEPFAKEBENCH_ROOT else None,
)
CHECKPOINT = _evaluation_path(
    "VERIFRAME_XCEPTION_EVALUATION_CHECKPOINT",
    DEEPFAKEBENCH_ROOT / "custom_training_source_disjoint"
    / "xception_custom_best.pth"
    if DEEPFAKEBENCH_ROOT else None,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = BACKEND_ROOT / "runtime" / "model_evaluation"


def validate_evaluation_files() -> None:
    """Check that the isolated model environment and evaluation inputs exist."""
    required = {
        "DeepfakeBench Python": MODEL_PYTHON,
        "Evaluation script": EVALUATION_SCRIPT,
        "Dataset CSV": DATASET_CSV,
        "Model checkpoint": CHECKPOINT,
    }

    missing = [
        f"{name}: not configured" if path is None else f"{name}: {path}"
        for name, path in required.items()
        if path is None or not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Model evaluation setup is incomplete:\n" + "\n".join(missing)
        )


def run_model_evaluation(
    split: str = "test",
    batch_size: int = 32,
    threshold: float = 0.5,
    timeout_seconds: int = 21600,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the existing batch evaluator in the DeepfakeBench environment.

    PyTorch is deliberately not imported into the FastAPI environment.
    """
    validate_evaluation_files()

    if split not in {"train", "val", "validation", "test"}:
        raise ValueError("Unsupported split name.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    evaluation_id = evaluation_id or str(uuid.uuid4())
    output_dir = EVALUATION_ROOT / evaluation_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"xception_{split}_summary.json"

    command = [
        str(MODEL_PYTHON),
        str(EVALUATION_SCRIPT),
        "--csv",
        str(DATASET_CSV),
        "--checkpoint",
        str(CHECKPOINT),
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--threshold",
        str(threshold),
        "--output-dir",
        str(output_dir),
    ]

    started_at = datetime.now(timezone.utc).isoformat()

    try:
        completed = subprocess.run(
            command,
            cwd=str(DEEPFAKEBENCH_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "DeepfakeBench evaluation failed.\n"
                f"Exit code: {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout[-12000:]}\n"
                f"STDERR:\n{completed.stderr[-12000:]}"
            )

        if not summary_path.is_file():
            raise RuntimeError(
                "Evaluation process finished, but the summary JSON was not created."
            )

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        return {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "created_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": "Custom DeepfakeBench Xception",
            "split": split,
            "batch_size": batch_size,
            "threshold": threshold,
            "summary": summary,
            "artifacts": {
                "summary_json": str(summary_path),
                "frame_csv": str(output_dir / f"xception_{split}_frame_results.csv"),
                "video_csv": str(output_dir / f"xception_{split}_video_results.csv"),
            },
            "stdout_tail": completed.stdout[-6000:],
            "stderr_tail": completed.stderr[-3000:],
        }

    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Model evaluation exceeded {timeout_seconds} seconds."
        ) from exc

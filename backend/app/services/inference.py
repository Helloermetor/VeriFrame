from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

# ---------------- SHARED PATH CONFIGURATION ----------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = "Scripts" if os.name == "nt" else "bin"
PYTHON_EXE = "python.exe" if os.name == "nt" else "python"

def _environment_python(variable: str, environment: str) -> Path:
    override = os.getenv(variable)
    if override:
        return Path(override).expanduser()
    return BACKEND_ROOT / environment / VENV_BIN / PYTHON_EXE

# ---------------- XCEPTION CONFIGURATION ----------------
# Xception uses the source, checkpoint and environment bundled with VeriFrame.
XCEPTION_ROOT = BACKEND_ROOT
XCEPTION_PYTHON = _environment_python("VERIFRAME_XCEPTION_PYTHON", ".venv")
XCEPTION_SCRIPT = Path(os.getenv(
    "VERIFRAME_XCEPTION_SCRIPT",
    str(XCEPTION_ROOT / "veriframe_xception_infer.py")
))
XCEPTION_CHECKPOINT = Path(os.getenv(
    "VERIFRAME_XCEPTION_CHECKPOINT",
    str(XCEPTION_ROOT.parent / "model_assets" / "Xception" / "xception_custom_best.pth")
))

# ---------------- GEND CONFIGURATION ----------------
GEND_ROOT = BACKEND_ROOT / "model_sources" / "gend"
GEND_PYTHON = _environment_python("VERIFRAME_GEND_PYTHON", ".venv-gend")
GEND_SCRIPT = Path(os.getenv("VERIFRAME_GEND_SCRIPT", str(Path(__file__).resolve().parents[2] / "gend_infer.py")))

# ---------------- EFFORT CONFIGURATION ----------------
EFFORT_ROOT = BACKEND_ROOT / "model_sources" / "effort"
EFFORT_PYTHON = _environment_python("VERIFRAME_EFFORT_PYTHON", ".venv-gend")
EFFORT_SCRIPT = Path(os.getenv("VERIFRAME_EFFORT_SCRIPT", str(Path(__file__).resolve().parents[1] / "effort_infer.py")))
EFFORT_CHECKPOINT = Path(os.getenv("VERIFRAME_EFFORT_CHECKPOINT", str(BACKEND_ROOT / "model_assets" / "Effort" / "effort_clip_L14_trainOn_FaceForensic.pth")))

# ---------------- FAKESTORMER CONFIGURATION ----------------
FAKESTORMER_ROOT = BACKEND_ROOT / "model_sources" / "fakestormer"
FAKESTORMER_PYTHON = _environment_python("VERIFRAME_FAKESTORMER_PYTHON", ".venv-fakestormer")
FAKESTORMER_SCRIPT = Path(os.getenv("VERIFRAME_FAKESTORMER_SCRIPT", str(Path(__file__).resolve().parents[1] / "fakestormer_infer.py")))

# ---------------- SHARED SETTINGS ----------------
MAX_FRAMES = max(1, int(os.getenv("VERIFRAME_MAX_FRAMES", "32")))
SAMPLE_FPS = max(0.1, float(os.getenv("VERIFRAME_SAMPLE_FPS", "1.0")))
TIMEOUT_SECONDS = max(60, int(os.getenv("VERIFRAME_INFERENCE_TIMEOUT_SECONDS", "1800")))


def _run_json_command(command: list[str], cwd: Path, model_name: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"{model_name} exceeded {TIMEOUT_SECONDS} seconds."}
    except OSError as exc:
        return {"status": "error", "error": f"Could not launch {model_name}: {exc}"}

    payload = None
    for line in reversed((completed.stdout or "").splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

    if payload is None:
        detail = (completed.stderr or completed.stdout or "No JSON output returned.")[-3000:]
        return {"status": "error", "error": f"{model_name} returned invalid JSON: {detail}"}
    if completed.returncode != 0 or payload.get("status") != "ok":
        return {
            "status": "error",
            "error": payload.get("error", f"{model_name} process failed."),
            "error_type": payload.get("error_type"),
        }
    return payload


def run_gend(video_path: str) -> dict[str, Any]:
    for path, description in [
        (GEND_PYTHON, "GenD Python interpreter"),
        (GEND_SCRIPT, "GenD inference script"),
    ]:
        if not path.is_file():
            return {"status": "error", "error": f"{description} not found: {path}"}
    command = [
        str(GEND_PYTHON), str(GEND_SCRIPT), str(Path(video_path).resolve()),
        "--max-frames", str(MAX_FRAMES), "--sample-fps", str(SAMPLE_FPS),
    ]
    return _run_json_command(command, GEND_ROOT, "GenD")


def run_xception(video_path: str) -> dict[str, Any]:
    for path, description in [
        (XCEPTION_PYTHON, "DeepfakeBench Python interpreter"),
        (XCEPTION_SCRIPT, "Xception inference script"),
        (XCEPTION_CHECKPOINT, "Xception checkpoint"),
    ]:
        if not path.is_file():
            return {"status": "error", "error": f"{description} not found: {path}"}
    command = [
        str(XCEPTION_PYTHON), str(XCEPTION_SCRIPT), str(Path(video_path).resolve()),
        "--checkpoint", str(XCEPTION_CHECKPOINT), "--max-frames", str(MAX_FRAMES),
        "--sample-fps", str(SAMPLE_FPS),
    ]
    return _run_json_command(command, XCEPTION_ROOT, "Xception")


def run_effort(video_path: str) -> dict[str, Any]:
    for path, description in [
        (EFFORT_PYTHON, "Effort Python interpreter"),
        (EFFORT_SCRIPT, "Effort inference script"),
        (EFFORT_CHECKPOINT, "Effort checkpoint"),
    ]:
        if not path.is_file():
            return {"status": "error", "error": f"{description} not found: {path}"}
    command = [
        str(EFFORT_PYTHON), str(EFFORT_SCRIPT), str(Path(video_path).resolve()),
        "--max-frames", str(MAX_FRAMES), "--sample-fps", str(SAMPLE_FPS),
    ]
    return _run_json_command(command, EFFORT_ROOT, "Effort")


def run_fakestormer(video_path: str) -> dict[str, Any]:
    for path, description in [
        (FAKESTORMER_PYTHON, "FakeSTormer Python interpreter"),
        (FAKESTORMER_SCRIPT, "FakeSTormer inference script"),
    ]:
        if not path.is_file():
            return {"status": "error", "error": f"{description} not found: {path}"}
    command = [
        str(FAKESTORMER_PYTHON),
        str(FAKESTORMER_SCRIPT),
        str(Path(video_path).resolve()),
    ]
    return _run_json_command(command, FAKESTORMER_ROOT, "FakeSTormer")


def _model_status(model_id: str, role: str, payload: dict[str, Any]) -> dict[str, Any]:
    successful = payload.get("status") == "ok"
    result = {
        "id": model_id,
        "role": role,
        "status": "completed" if successful else "failed",
    }
    if not successful:
        result["reason"] = payload.get("error", "Inference failed.")
    return result


def run_committee(video_path: str) -> dict[str, Any]:
    """Run all four detectors sequentially in independent Python environments."""

    # Sequential execution limits competing GPU memory allocations.
    gend = run_gend(video_path)
    xception = run_xception(video_path)
    effort = run_effort(video_path)
    fakestormer = run_fakestormer(video_path)

    results = {
        "gend": gend,
        "xception": xception,
        "effort": effort,
        "fakestormer": fakestormer,
    }
    successful = {key: value.get("status") == "ok" for key, value in results.items()}
    completed_count = sum(successful.values())

    if completed_count == 4:
        inference_status, mode = "completed", "gend_xception_effort_fakestormer"
    elif completed_count > 0:
        inference_status, mode = "partial", "partial_committee"
    else:
        inference_status, mode = "failed", "no_model_available"

    models = [
        _model_status("gend_clip_l14", "Primary detector: GenD CLIP ViT-L/14", gend),
        _model_status("xception_custom", "Control detector: custom DeepfakeBench Xception", xception),
        _model_status("effort", "Challenger 1: Effort CLIP ViT-L/14 with SVD adaptation", effort),
        _model_status("fakestormer", "Challenger 2: FakeSTormer C23 ViT-Base", fakestormer),
    ]

    # FakeSTormer temporal-localization outputs are intentionally not added
    # to frame_scores, which contains classification scores from other models.
    frame_scores = {
        "gend_clip_l14": gend.get("frames", []) if successful["gend"] else [],
        "xception_custom": xception.get("frames", []) if successful["xception"] else [],
        "effort": effort.get("frames", []) if successful["effort"] else [],
    }

    errors = {
        key: None if successful[key] else results[key].get("error")
        for key in results
    }
    successful_names = [
        name
        for name, key in [
            ("GenD", "gend"),
            ("Xception", "xception"),
            ("Effort", "effort"),
            ("FakeSTormer", "fakestormer"),
        ]
        if successful[key]
    ]

    if completed_count == 4:
        reason = (
            "All four detectors completed independently. Scores remain uncalibrated, "
            "and FakeSTormer temporal-localization outputs are retained separately. "
            "The overall forensic verdict must remain INCONCLUSIVE."
        )
    elif completed_count > 0:
        reason = (
            f"Only {completed_count} of 4 configured detectors completed "
            f"({', '.join(successful_names)}). Results are partial, and "
            "the forensic verdict remains INCONCLUSIVE."
        )
    else:
        reason = (
            "No detector completed successfully. "
            + " ".join(f"{key}: {errors[key]}" for key in errors if errors[key])
        )

    return {
        "inference_status": inference_status,
        "mode": mode,
        "reason": reason,
        "models": models,
        "gend": gend if successful["gend"] else None,
        "xception": xception if successful["xception"] else None,
        "effort": effort if successful["effort"] else None,
        "fakestormer": fakestormer if successful["fakestormer"] else None,
        "model_errors": errors,
        "frame_scores": frame_scores,
        "calibrated_probability": None,
        "conformal_set": None,
    }


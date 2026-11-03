from __future__ import annotations

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any

from .inference import run_committee

FFMPEG_PATH = (
    os.getenv("VERIFRAME_FFMPEG_PATH")
    or shutil.which("ffmpeg")
)


def _extract_score_and_output(
    committee: dict[str, Any],
) -> tuple[float | None, str]:
    """Extract the aggregate Xception score and model output."""
    xception = committee.get("xception") or {}

    score = xception.get("class1_manipulated_score_mean")
    output = xception.get("threshold_model_output", "unavailable")

    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None

    return score, str(output or "unavailable")


def _run_ffmpeg(
    source: str,
    destination: str,
    filters: list[str],
    extra_args: list[str] | None = None,
) -> None:
    """Create a transformed video using FFmpeg."""
    if not FFMPEG_PATH or not Path(FFMPEG_PATH).is_file():
        raise FileNotFoundError(
            "FFmpeg executable was not found on PATH. "
            "Install FFmpeg or set VERIFRAME_FFMPEG_PATH."
        )

    command = [
        str(FFMPEG_PATH),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]

    if filters:
        command.extend(["-vf", ",".join(filters)])

    if extra_args:
        command.extend(extra_args)

    command.extend(["-c:a", "copy", destination])

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "Unknown FFmpeg error").strip()

        raise RuntimeError(f"FFmpeg transformation failed: {message[-1200:]}")


def run_stability(video_path: str) -> dict[str, Any]:
    """
    Compare Xception scores on the original video and five transformed
    versions. Results are diagnostic and do not measure model accuracy.
    """
    baseline_committee = run_committee(video_path)

    baseline_score, baseline_output = _extract_score_and_output(baseline_committee)

    score_definition = (
        "Mean absolute difference between baseline and perturbed video "
        "Xception scores. Lower means more stable under these specific "
        "tests; it does not measure detection accuracy."
    )

    if baseline_score is None:
        return {
            "status": "failed",
            "baseline_score": None,
            "baseline_output": baseline_output,
            "score": None,
            "score_definition": score_definition,
            "decision_flip_count": None,
            "completed_test_count": 0,
            "requested_test_count": 5,
            "perturbations": [],
            "diagnostic_only": True,
            "error": (
                "Baseline Xception score was unavailable; "
                "stability tests were not run."
            ),
        }

    # Encoding settings for the transformed videos.
    encoding_args = [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    tests = [
        {
            "id": "brightness",
            "label": "Mild brightness shift",
            "filters": ["eq=brightness=0.04"],
            "extra_args": encoding_args,
        },
        {
            "id": "resize",
            "label": "Resize and restore",
            "filters": [
                "scale=iw/2:ih/2",
                "scale=iw*2:ih*2",
            ],
            "extra_args": encoding_args,
        },
        {
            "id": "jpeg_compression",
            "label": "Video compression (H.264 CRF 35)",
            "filters": [],
            "extra_args": [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "35",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        },
        {
            "id": "gaussian_blur",
            "label": "Mild Gaussian blur",
            "filters": ["gblur=sigma=1.0"],
            "extra_args": encoding_args,
        },
        {
            "id": "gaussian_noise",
            "label": "Light Gaussian noise",
            "filters": ["noise=alls=5:allf=t+u"],
            "extra_args": encoding_args,
        },
    ]

    perturbations: list[dict[str, Any]] = []
    score_deltas: list[float] = []
    decision_flip_count = 0

    with tempfile.TemporaryDirectory(prefix="veriframe_stability_") as temp_dir:
        for test in tests:
            destination = str(Path(temp_dir) / f"{test['id']}.mp4")

            item: dict[str, Any] = {
                "id": test["id"],
                "label": test["label"],
                "status": "failed",
                "score": None,
                "absolute_score_delta": None,
                "signed_score_delta": None,
                "decision_changed": None,
            }

            try:
                _run_ffmpeg(
                    video_path,
                    destination,
                    test["filters"],
                    test["extra_args"],
                )

                committee = run_committee(destination)

                perturbed_score, perturbed_output = _extract_score_and_output(committee)

                if perturbed_score is None:
                    raise RuntimeError(
                        "Xception score was unavailable for " "this transformed video."
                    )

                signed_delta = perturbed_score - baseline_score
                absolute_delta = abs(signed_delta)

                decision_changed = perturbed_output != baseline_output

                item.update(
                    {
                        "status": "completed",
                        "score": round(perturbed_score, 6),
                        "output": perturbed_output,
                        "absolute_score_delta": round(absolute_delta, 6),
                        "signed_score_delta": round(signed_delta, 6),
                        "decision_changed": decision_changed,
                    }
                )

                score_deltas.append(absolute_delta)

                if decision_changed:
                    decision_flip_count += 1

            except Exception as exc:
                item["reason"] = str(exc)

            perturbations.append(item)

    completed_count = len(score_deltas)

    mean_absolute_delta = (
        sum(score_deltas) / completed_count if completed_count else None
    )

    if completed_count == len(tests):
        status = "completed"
    elif completed_count > 0:
        status = "partial"
    else:
        status = "failed"

    return {
        "status": status,
        "baseline_score": round(baseline_score, 6),
        "baseline_output": baseline_output,
        "score": (
            round(mean_absolute_delta, 6) if mean_absolute_delta is not None else None
        ),
        "score_definition": score_definition,
        "decision_flip_count": decision_flip_count,
        "completed_test_count": completed_count,
        "requested_test_count": len(tests),
        "perturbations": perturbations,
        "diagnostic_only": True,
    }

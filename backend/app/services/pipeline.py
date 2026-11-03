from __future__ import annotations

import hashlib
import json
import os
import subprocess
import shutil
from pathlib import Path

from . import store
from .inference import run_committee

UPLOAD_DIR = Path(os.getenv("VERIFRAME_UPLOAD_DIR", "./runtime/uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FFPROBE_PATH = (
    os.getenv("VERIFRAME_FFPROBE_PATH")
    or shutil.which("ffprobe")
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_metadata(path: str) -> dict:
    try:
        proc = subprocess.run(
            [
                FFPROBE_PATH or "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        data = json.loads(proc.stdout)
        fmt = data.get("format", {})
        video = next(
            (
                stream
                for stream in data.get("streams", [])
                if stream.get("codec_type") == "video"
            ),
            {},
        )
        return {
            "container": fmt.get("format_name"),
            "duration_seconds": (
                float(fmt["duration"]) if fmt.get("duration") else None
            ),
            "size_bytes": (
                int(fmt["size"])
                if fmt.get("size")
                else Path(path).stat().st_size
            ),
            "video_codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "metadata_status": "available",
        }
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        ValueError,
        KeyError,
    ):
        return {
            "container": None,
            "duration_seconds": None,
            "size_bytes": Path(path).stat().st_size,
            "video_codec": None,
            "width": None,
            "height": None,
            "metadata_status": "ffprobe_unavailable_or_failed",
        }


def _valid_score(model: dict):
    value = model.get("class1_manipulated_score_mean")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _model_output(model: dict):
    score = _valid_score(model)
    if score is None:
        return None

    threshold = model.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.5

    return "manipulated" if score >= threshold else "original"



def _load_calibration_parameters() -> dict:
    calibration_path = (
        Path(__file__).resolve().parents[3]
        / "validation"
        / "calibration_test_split"
        / "calibration_parameters.json"
    )
    with calibration_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _calibrate_score(score, calibration: dict):
    if score is None:
        return None

    import math

    try:
        score = min(max(float(score), 1e-6), 1 - 1e-6)
        coefficient = float(calibration["coefficient"])
        intercept = float(calibration["intercept"])
        logit = math.log(score / (1 - score))
        value = coefficient * logit + intercept

        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

def _build_frame_rows(committee: dict) -> list:
    scores = committee.get("frame_scores") or {}

    # Current inference format: separate frame lists for each model.
    if isinstance(scores, dict):
        rows = []
        for model_id, model_frames in scores.items():
            if not isinstance(model_frames, list):
                continue
            for frame in model_frames:
                if isinstance(frame, dict):
                    rows.append({"model_id": model_id, **frame})
        return rows

    # Backward compatibility with the earlier single-model list format.
    if isinstance(scores, list):
        return [
            {"model_id": "xception_custom", **frame}
            for frame in scores
            if isinstance(frame, dict)
        ]

    return []


def run_pipeline(analysis_id: str, video_path: str) -> None:
    try:
        store.add_event(analysis_id, "validating", 10, "Upload validated")
        metadata = extract_metadata(video_path)

        store.add_event(analysis_id, "metadata", 20, "Media metadata inspected")
        store.add_event(analysis_id, "sampling", 35, "Preparing sampled video frames")
        store.add_event(analysis_id, "inference", 65,
                        "Running GenD, custom Xception, Effort, and FakeSTormer inference")

        committee = run_committee(video_path)

        # Calibrated primary decision: GenD + custom Xception.
        calibration_parameters = _load_calibration_parameters()
        calibrated_probabilities = {}

        for model_key in ("gend", "xception"):
            model = committee.get(model_key) or {}
            calibration = calibration_parameters["calibrated_models"].get(model_key)
            raw_score = _valid_score(model) if model.get("status") == "ok" else None
            calibrated_probabilities[model_key] = (
                _calibrate_score(raw_score, calibration) if calibration else None
            )

        gend_probability = calibrated_probabilities["gend"]
        xception_probability = calibrated_probabilities["xception"]

        if gend_probability is not None and xception_probability is not None:
            gend_decision = "FAKE" if gend_probability >= 0.5 else "REAL"
            xception_decision = "FAKE" if xception_probability >= 0.5 else "REAL"
            outcome = (
                gend_decision
                if gend_decision == xception_decision
                else "INCONCLUSIVE"
            )
        else:
            outcome = "INCONCLUSIVE"

        committee["calibrated_probabilities"] = calibrated_probabilities
        model_specs = [
            ("gend_clip_l14", "GenD CLIP ViT-L/14", "gend"),
            ("xception_custom", "Custom DeepfakeBench Xception", "xception"),
            ("effort", "Effort CLIP ViT-L/14 with SVD adaptation", "effort"),
        ]
        successful_models = []
        model_ok = {}
        for model_id, display_name, committee_key in model_specs:
            model = committee.get(committee_key) or {}
            ok = model.get("status") == "ok"
            model_ok[committee_key] = ok
            if ok:
                successful_models.append((model_id, display_name, model))

        # FakeSTormer is tracked as a fourth detector, but its temporal-localization
        # outputs are not class-1 manipulated scores and are excluded from score voting.
        fakestormer = committee.get("fakestormer") or {}
        fakestormer_ok = fakestormer.get("status") == "ok"
        model_ok["fakestormer"] = fakestormer_ok
        successful_detector_count = len(successful_models) + int(fakestormer_ok)
        inference_ok = successful_detector_count > 0
        frame_rows = _build_frame_rows(committee)
        metadata_source = next((committee.get(key) for key in ("gend", "xception", "effort", "fakestormer")
                                if model_ok.get(key) and committee.get(key)), {})
        if inference_ok:
            for field in ("width", "height", "duration_seconds"):
                if metadata.get(field) is None:
                    metadata[field] = metadata_source.get(field)
            metadata["sampled_frame_count"] = metadata_source.get("sampled_frame_count")
            metadata["source_fps"] = metadata_source.get("source_fps")

        file_hash = (store.get(analysis_id) or {}).get("sha256", "unavailable")
        model_evidence = []
        contradictory = []
        model_outputs = {}
        for model_id, display_name, model in successful_models:
            score = _valid_score(model)
            output = _model_output(model)
            model_outputs[model_id] = output
            model_evidence.extend([
                {"label": f"{display_name} manipulated-class score (uncalibrated)",
                 "value": score,
                 "detail": "Mean of sampled-frame class-1 scores. Not a calibrated probability."},
                {"label": f"{display_name} threshold-based model output", "value": output or "unavailable"},
                {"label": f"{display_name} sampled frames", "value": model.get("sampled_frame_count")},
                {"label": f"{display_name} device", "value": model.get("gpu_name") or model.get("device")},
                {"label": f"{display_name} checkpoint SHA-256", "value": model.get("checkpoint_sha256")},
            ])

        # Preserve FakeSTormer evidence in its native temporal-localization format.
        # Do not add its official label to the class-score agreement calculation.
        if fakestormer_ok:
            model_evidence.extend([
                {"label": "FakeSTormer temporal-localization values (uncalibrated)",
                 "value": fakestormer.get("temporal_localization_values"),
                 "detail": fakestormer.get("score_semantics") or
                           "Raw sigmoid temporal-localization outputs; not calibrated probabilities."},
                {"label": "FakeSTormer final temporal-localization value",
                 "value": fakestormer.get("temporal_localization_final")},
                {"label": "FakeSTormer official threshold-based label",
                 "value": fakestormer.get("official_label"),
                 "detail": "Model-reported label from the official FakeSTormer test_vid threshold rule; not the VeriFrame verdict."},
                {"label": "FakeSTormer threshold", "value": fakestormer.get("threshold")},
                {"label": "FakeSTormer classification-head outputs (uncalibrated)",
                 "value": fakestormer.get("classification_head_values"),
                 "detail": "Raw sigmoid head outputs; not calibrated probabilities."},
                {"label": "FakeSTormer sampled frames", "value": fakestormer.get("sampled_frame_count")},
                {"label": "FakeSTormer device", "value": fakestormer.get("gpu_name") or fakestormer.get("device")},
                {"label": "FakeSTormer inference time (seconds)", "value": fakestormer.get("inference_seconds")},
                {"label": "FakeSTormer checkpoint SHA-256", "value": fakestormer.get("checkpoint_sha256")},
            ])

        # Report agreement/disagreement only when at least two class-score outputs exist.
        available_outputs = [(model_id, output) for model_id, output in model_outputs.items() if output is not None]
        if len(available_outputs) >= 2:
            distinct_outputs = {output for _, output in available_outputs}
            names = {"gend_clip_l14": "GenD", "xception_custom": "Xception", "effort": "Effort"}
            if len(distinct_outputs) == 1:
                contradictory.append("All completed detectors agree on their threshold-based output: "
                                     f"{available_outputs[0][1]}. This is model agreement, not a validated forensic conclusion.")
            else:
                groups = {}
                for model_id, output in available_outputs:
                    groups.setdefault(output, []).append(names.get(model_id, model_id))
                summary = "; ".join(f"{', '.join(group)}: {label}" for label, group in groups.items())
                contradictory.append("Detectors disagree on their threshold-based outputs (" + summary + "). "
                                     "This disagreement requires further investigation.")
        elif successful_models:
            contradictory.append("Fewer than two completed detectors produced threshold-based outputs; "
                                 "independent comparison is unavailable.")
        else:
            contradictory.append("No valid model output is available; the system cannot support an authenticity claim.")

        if successful_models:
            score_summary = "; ".join(
                f"{display_name}: {_valid_score(model):.6f}"
                for _, display_name, model in successful_models
                if _valid_score(model) is not None
            )
            completion_detail = f"{successful_detector_count} of 4 detectors completed."
            if score_summary:
                outcome_note = (
                    f"{completion_detail} Class-score outputs: {score_summary}. "
                    "All detector outputs are uncalibrated and are not validated probabilities. "
                    "FakeSTormer temporal-localization evidence is reported separately. "
                    "The overall forensic outcome remains INCONCLUSIVE."
                )
            else:
                outcome_note = (
                    f"{completion_detail} FakeSTormer temporal-localization evidence is available, "
                    "but no class-score output is available for comparison. "
                    "The overall forensic outcome remains INCONCLUSIVE."
                )
            status_message = (
                f"{completion_detail} Forensic verdict remains inconclusive"
            )
        else:
            reason = committee.get("reason", "No detector returned a valid result.")
            outcome_note = f"No model score is available. {reason}"
            status_message = "Analysis finished without a valid model score"

        limitations = [
            "The verdict uses agreement between calibrated GenD and custom Xception scores. Effort remains supporting evidence; FakeSTormer temporal-localization outputs are reported separately. Calibration and independent-test performance do not guarantee generalization to unseen manipulation methods.",
            "C2PA manifest validation, face tracking, face-quality gating, Grad-CAM faithfulness, and full temporal analysis are not yet implemented.",
            "The Stability Panel provides diagnostic Xception transformation tests; comprehensive robustness validation has not been established.",
            "The current prototype has not been validated for fully AI-generated video.",
            "Only the largest detected face is processed by the current GenD inference implementation.",
        ]
        stale_or_duplicate_terms = (
            "single custom xception detector only", "gend clip/dinov3", "dinov3", "xdfdet",
            "not connected", "only one detector", "stability panel", "robustness testing",
            "robustness checks", "robustness validation", "overall verdict must remain inconclusive",
            "verdict must remain inconclusive", "until validation and calibration are completed",
            "overall forensic outcome should remain inconclusive", "overall outcome should remain inconclusive",
            "overall forensic verdict is held at inconclusive", "class-1 scores are uncalibrated",
            "class 1 scores are uncalibrated", "gend scores are uncalibrated",
            "xception scores are uncalibrated", "effort scores are uncalibrated", "scores are uncalibrated",
            "no face-quality gating, tracking", "no face tracking or quality gating",
            "only the largest detected face is processed",
        )
        normalized_limitations = {item.casefold() for item in limitations}
        for _, _, model in successful_models:
            for limitation in model.get("limitations", []):
                if not isinstance(limitation, str) or not limitation.strip():
                    continue
                normalized = limitation.casefold().strip()
                if any(term in normalized for term in stale_or_duplicate_terms):
                    continue
                if normalized not in normalized_limitations:
                    limitations.append(limitation)
                    normalized_limitations.add(normalized)

        error_labels = {"gend": "GenD", "xception": "Xception", "effort": "Effort",
                        "fakestormer": "FakeSTormer"}
        for key, label in error_labels.items():
            if not model_ok.get(key):
                error = (committee.get("model_errors") or {}).get(key) or "not available"
                limitations.append(f"{label} did not complete: {error}")

        # Publish calibrated primary-model evidence and the final decision.
        for model_key, display_name in (
            ("gend", "GenD"),
            ("xception", "Xception"),
        ):
            probability = calibrated_probabilities.get(model_key)
            model_evidence.append({
                "label": f"{display_name} calibrated manipulated probability",
                "value": probability,
                "detail": "Platt-scaled using the frozen calibration parameters.",
            })

        if gend_probability is not None and xception_probability is not None:
            outcome_note = (
                f"GenD calibrated manipulated probability: {gend_probability:.4f}. "
                f"Xception calibrated manipulated probability: {xception_probability:.4f}. "
                + (
                    f"Both primary models agree on {outcome}."
                    if outcome != "INCONCLUSIVE"
                    else "The primary models disagree; no definitive verdict is issued."
                )
            )
        else:
            outcome_note = (
                "A definitive verdict requires calibrated scores from both GenD and Xception. "
                "At least one primary model did not provide a valid score."
            )

        status_message = (
            f"Analysis completed. VeriFrame verdict: {outcome}"
        )

        result = {
            "outcome": outcome,
            "outcome_note": outcome_note,
            "metadata": metadata,
            "committee": committee,
            "evidence": {
                "contradictory_evidence": contradictory,
                "model_evidence": model_evidence,
                "provenance_evidence": [
                    {"label": "Original upload SHA-256", "value": file_hash},
                    {"label": "C2PA", "value": "not_checked"},
                    {"label": "Media metadata", "value": metadata["metadata_status"]},
                ],
                "quality_evidence": [
                    {"label": "Face visibility / tracking", "value": "not_measured"},
                    {"label": "Blur / occlusion quality gates", "value": "not_measured"},
                ],
                "limitations": limitations,
            },
            "frames": frame_rows,
            "stability": {"status": "not_run", "score": None, "perturbations": []},
        }
        store.update(analysis_id, result=result, status="completed", progress=100, stage="completed")
        store.add_event(analysis_id, "completed", 100, status_message)

    except Exception as exc:
        store.update(
            analysis_id,
            status="failed",
            error=str(exc),
            stage="failed",
        )
        store.add_event(analysis_id, "failed", 100, "Analysis failed")

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

GEND_ROOT = Path(__file__).resolve().parent / "model_sources" / "gend"
MODEL_ID = "yermandy/GenD_CLIP_L_14"
MODEL_REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"

MODEL_DIR = Path(__file__).resolve().parent / "model_assets" / "GenD"
CLIP_DIR = MODEL_DIR / "clip-vit-large-patch14"
CHECKPOINT = MODEL_DIR / "model.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(payload: dict, exit_code: int = 0) -> int:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return exit_code


def run(video_path: Path, max_frames: int, sample_fps: float) -> dict:
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    os.chdir(GEND_ROOT)
    sys.path.insert(0, str(GEND_ROOT))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from src.retinaface import prepare_model
    from detector import align_face
    from src.hf.modeling_gend import GenD, GenDConfig

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"GenD checkpoint not found: {CHECKPOINT}")

    if max_frames < 1 or sample_fps <= 0:
        raise ValueError("Invalid frame sampling configuration.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    required_clip_files = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
    ]

    if not CLIP_DIR.is_dir():
        raise FileNotFoundError(f"CLIP backbone directory not found: {CLIP_DIR}")

    missing_clip_files = [
        name for name in required_clip_files
        if not (CLIP_DIR / name).is_file()
    ]

    if missing_clip_files:
        raise FileNotFoundError(
            f"CLIP backbone files missing: {missing_clip_files}"
        )

    config = GenDConfig.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    config.backbone = str(CLIP_DIR)

    model = GenD.from_pretrained(
        str(MODEL_DIR),
        config=config,
        local_files_only=True,
    ).to(device).eval()

    face_detector = prepare_model(det_thres=0.1)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    frames = []
    skipped = []

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if total_frames <= 0:
            raise RuntimeError("Video contains no readable frames.")

        if fps <= 0:
            fps = 25.0

        stride = max(1, int(round(fps / sample_fps)))
        candidates = list(range(0, total_frames, stride))

        if len(candidates) > max_frames:
            positions = np.linspace(
                0, len(candidates) - 1, num=max_frames, dtype=int
            )
            candidates = [candidates[int(p)] for p in positions]

        for frame_idx in sorted(set(candidates)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = capture.read()

            if not ok or bgr is None:
                skipped.append({
                    "frame_index": int(frame_idx),
                    "reason": "decode_failed",
                })
                continue

            boxes, landmarks = face_detector.detect(bgr)

            if boxes is None or len(boxes) == 0:
                skipped.append({
                    "frame_index": int(frame_idx),
                    "reason": "no_face_detected",
                })
                continue

            areas = (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            )
            idx = int(np.argmax(areas))

            aligned, _ = align_face(
                bgr,
                landmarks[idx],
                target_size=None,
                scale=1.3,
            )

            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            tensor = model.feature_extractor.preprocess(image)

            frames.append({
                "frame_index": int(frame_idx),
                "timestamp_seconds": round(frame_idx / fps, 3),
                "tensor": tensor,
            })

    finally:
        capture.release()

    if not frames:
        raise RuntimeError("No faces could be detected and aligned.")

    scores = []

    with torch.inference_mode():
        for item in frames:
            tensor = item["tensor"].unsqueeze(0).to(device)
            logits = model(tensor)

            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(
                    f"Unexpected GenD output shape: {tuple(logits.shape)}"
                )

            probabilities = torch.softmax(logits, dim=1)[0]
            fake_score = float(probabilities[1].item())
            scores.append(fake_score)

    frame_rows = [
        {
            "frame_index": item["frame_index"],
            "timestamp_seconds": item["timestamp_seconds"],
            "class1_manipulated_score": round(score, 6),
            "threshold_model_output": (
                "manipulated" if score >= 0.5 else "original"
            ),
        }
        for item, score in zip(frames, scores)
    ]

    return {
        "status": "ok",
        "model_id": "gend_clip_l14",
        "model_name": "GenD CLIP ViT-L/14",
        "model_revision": MODEL_REVISION,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda" else None
        ),
        "checkpoint_filename": CHECKPOINT.name,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "class_mapping": {
            "0": "original",
            "1": "manipulated",
        },
        "score_definition": (
            "Mean of per-frame GenD class-1 softmax scores; "
            "uncalibrated model score, not an authenticity probability."
        ),
        "score_is_calibrated_probability": False,
        "threshold": 0.5,
        "class1_manipulated_score_mean": round(
            float(np.mean(scores)), 6
        ),
        "class1_manipulated_score_median": round(
            float(np.median(scores)), 6
        ),
        "class1_manipulated_score_std": round(
            float(np.std(scores)), 6
        ),
        "sampled_frame_count": len(scores),
        "requested_frame_count": len(candidates),
        "face_skipped_frame_count": len(skipped),
        "face_skipped_frames": skipped,
        "source_frame_count": total_frames,
        "source_fps": round(fps, 4),
        "duration_seconds": round(total_frames / fps, 3),
        "width": width,
        "height": height,
        "frames": frame_rows,
        "preprocessing": (
            "GenD RetinaFace detection threshold 0.1; "
            "five-landmark alignment; target_size=None; scale=1.3; "
            "official GenD feature extractor."
        ),
        "limitations": [
            "GenD scores are uncalibrated.",
            "Only the largest detected face is processed per sampled frame.",
            "No face tracking or quality gating is implemented.",
            "The overall VeriFrame verdict must remain INCONCLUSIVE "
            "until validation and calibration are completed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    args = parser.parse_args()

    try:
        payload = run(
            args.video.resolve(),
            args.max_frames,
            args.sample_fps,
        )
        return emit(payload)
    except Exception as exc:
        return emit({
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        })


if __name__ == "__main__":
    raise SystemExit(main())

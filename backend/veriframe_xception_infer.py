"""Run the custom DeepfakeBench Xception checkpoint on aligned face crops.

Uses the same DeepfakeBench dlib face alignment/cropping helper as preprocessing,
then the same 256x256 RGB and Normalize(0.5, 0.5) transforms as training.
Outputs one JSON object for the VeriFrame subprocess bridge.
Scores are uncalibrated; they are not authenticity probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "model_sources" / "xception"
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "model_assets"
    / "Xception"
    / "xception_custom_best.pth"
)
PREDICTOR_PATH = (
    ROOT / "preprocessing" / "dlib_tools" / "shape_predictor_81_face_landmarks.dat"
)
IMAGE_SIZE = 256
THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(payload: dict, exit_code: int = 0) -> int:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return exit_code


def run(video_path: Path, checkpoint: Path, max_frames: int, sample_fps: float) -> dict:
    import cv2
    import numpy as np
    import torch
    import dlib
    from PIL import Image
    from torchvision import transforms

    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Xception checkpoint does not exist: {checkpoint}")
    if not PREDICTOR_PATH.is_file():
        raise FileNotFoundError(
            f"Dlib landmark predictor not found: {PREDICTOR_PATH}. "
            "The same predictor used by DeepfakeBench preprocessing is required."
        )
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")

    # Use DeepfakeBench's own alignment function, rather than resizing full frames.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "training"))
    from preprocessing.preprocess import extract_aligned_face_dlib
    from networks.xception import Xception

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Xception({"mode": "original", "num_classes": 2, "inc": 3, "dropout": False})
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not decode this video. Check the file/codec.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total_frames <= 0:
            raise RuntimeError(
                "Video reports zero frames or uses an unsupported codec."
            )
        if fps <= 0:
            fps = 25.0

        stride = max(1, int(round(fps / sample_fps)))
        candidate_indices = list(range(0, total_frames, stride)) or [0]
        if len(candidate_indices) > max_frames:
            positions = np.linspace(
                0, len(candidate_indices) - 1, num=max_frames, dtype=int
            )
            candidate_indices = [candidate_indices[int(pos)] for pos in positions]
        frame_indices = sorted(set(int(i) for i in candidate_indices))

        tensors = []
        frame_meta = []
        no_face_indices = []
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr_frame = capture.read()
            if not ok or bgr_frame is None:
                no_face_indices.append(
                    {"frame_index": frame_idx, "reason": "decode_failed"}
                )
                continue

            # extract_aligned_face_dlib expects a BGR frame and returns the
            # aligned 256x256 face crop in BGR (same function used in preprocessing).
            face_bgr, landmarks, _ = extract_aligned_face_dlib(
                face_detector, face_predictor, bgr_frame
            )
            if face_bgr is None or landmarks is None:
                no_face_indices.append(
                    {"frame_index": frame_idx, "reason": "no_face_or_landmarks"}
                )
                continue

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            tensors.append(transform(Image.fromarray(face_rgb)))
            frame_meta.append(
                {
                    "frame_index": int(frame_idx),
                    "timestamp_seconds": round(float(frame_idx) / fps, 3),
                }
            )
    finally:
        capture.release()

    if not tensors:
        raise RuntimeError(
            "No faces could be aligned in the sampled frames. The video may have no detectable faces, "
            "or the dlib predictor/codec setup may need checking. No full-frame fallback was used."
        )

    scores: list[float] = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            output = model(batch)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape[1] != 2
            ):
                raise RuntimeError(
                    f"Unexpected Xception output shape: {getattr(logits, 'shape', None)}"
                )
            class1 = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(float(value) for value in class1.detach().cpu().tolist())

    frames = [
        {
            **meta,
            "class1_manipulated_score": round(score, 6),
            "threshold_model_output": (
                "manipulated" if score >= THRESHOLD else "original"
            ),
        }
        for meta, score in zip(frame_meta, scores)
    ]
    mean_score = float(np.mean(scores))
    return {
        "status": "ok",
        "model_id": "xception_custom",
        "model_name": "Custom DeepfakeBench Xception (aligned face crops)",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_size": IMAGE_SIZE,
        "preprocessing": "DeepfakeBench dlib face detection + 5-point alignment/crop; RGB Resize(256,256); Normalize(mean=0.5,std=0.5)",
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "class_mapping": {"0": "original", "1": "manipulated"},
        "score_definition": "mean of per-face-crop softmax class-1 scores; uncalibrated model score, not an authenticity probability",
        "score_is_calibrated_probability": False,
        "threshold": THRESHOLD,
        "threshold_model_output": (
            "manipulated" if mean_score >= THRESHOLD else "original"
        ),
        "class1_manipulated_score_mean": round(mean_score, 6),
        "class1_manipulated_score_median": round(float(np.median(scores)), 6),
        "class1_manipulated_score_std": round(float(np.std(scores)), 6),
        "sampled_frame_count": len(scores),
        "requested_frame_count": len(frame_indices),
        "face_skipped_frame_count": len(no_face_indices),
        "face_skipped_frames": no_face_indices,
        "source_frame_count": total_frames,
        "source_fps": round(fps, 4),
        "duration_seconds": round(total_frames / fps, 3) if fps else None,
        "width": width,
        "height": height,
        "frames": frames,
        "limitations": [
            "Single custom Xception detector only; GenD CLIP/DINOv3 and xdfdet are not connected.",
            "Class-1 scores are uncalibrated and must not be interpreted as probabilities of authenticity.",
            "Face detection/alignment now follows the DeepfakeBench preprocessing helper, but the exact sampled frames may still differ from those used to create the training CSV.",
            "No face-quality gating, tracking, Grad-CAM faithfulness, robustness testing, calibration, or conformal prediction is implemented.",
            "The overall forensic outcome should remain INCONCLUSIVE until validation and robustness checks are complete.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    args = parser.parse_args()
    try:
        payload = run(
            args.video.resolve(),
            args.checkpoint.resolve(),
            args.max_frames,
            args.sample_fps,
        )
        return emit(payload, 0)
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, ModuleNotFoundError):
            missing = getattr(exc, "name", "unknown")
            hints = {
                "cv2": "Install OpenCV in DeepfakeBench .venv: python -m pip install opencv-python",
                "dlib": "The dlib package is required for the same DeepfakeBench face crop pipeline.",
                "imutils": "Install imutils in DeepfakeBench .venv: python -m pip install imutils",
                "skimage": "Install scikit-image in DeepfakeBench .venv: python -m pip install scikit-image",
            }
            message = hints.get(
                missing, f"Missing dependency '{missing}' in DeepfakeBench .venv: {exc}"
            )
        return emit(
            {"status": "error", "error": message, "error_type": type(exc).__name__}, 0
        )


if __name__ == "__main__":
    raise SystemExit(main())

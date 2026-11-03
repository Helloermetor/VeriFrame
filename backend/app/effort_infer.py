from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import yaml
from PIL import Image
from torchvision import transforms

BACKEND_ROOT = Path(__file__).resolve().parents[1]
EFFORT_ROOT = BACKEND_ROOT / "model_sources" / "effort"
CHECKPOINT = BACKEND_ROOT / "model_assets" / "Effort" / "effort_clip_L14_trainOn_FaceForensic.pth"
CONFIG = EFFORT_ROOT / "training" / "config" / "detector" / "effort.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Effort checkpoint not found: {CHECKPOINT}")
    if not CONFIG.is_file():
        raise FileNotFoundError(f"Effort config not found: {CONFIG}")

    os.chdir(EFFORT_ROOT)
    sys.path.insert(0, str(EFFORT_ROOT / "training"))
    from detectors import DETECTOR

    with CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETECTOR["effort"](config)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    checkpoint = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }
    result = model.load_state_dict(checkpoint, strict=False)
    allowed_missing = {
        key for key in model.state_dict()
        if key.endswith((".S_r", ".U_r", ".V_r"))
    }
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            "Effort checkpoint mismatch: "
            f"missing={result.missing_keys[:8]}, "
            f"unexpected={result.unexpected_keys[:8]}"
        )

    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / source_fps if source_fps > 0 and frame_count > 0 else None
    interval = max(1, round(source_fps / args.sample_fps)) if source_fps > 0 else 1

    frames = []
    frame_index = 0
    sampled_index = 0
    while len(frames) < max(1, args.max_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_index % interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model({"image": tensor})
                score = torch.softmax(output["cls"], dim=1)[0, 1].item()
            frames.append({
                "frame_index": frame_index,
                "sample_index": sampled_index,
                "timestamp_seconds": (
                    frame_index / source_fps if source_fps > 0 else None
                ),
                "manipulated_score": float(score),
            })
            sampled_index += 1
        frame_index += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames could be sampled from the video.")

    scores = [item["manipulated_score"] for item in frames]
    payload = {
        "status": "ok",
        "model_id": "effort",
        "model_name": "Effort CLIP ViT-L/14 with SVD residual adaptation",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "sampled_frame_count": len(frames),
        "source_fps": source_fps or None,
        "duration_seconds": duration,
        "class1_manipulated_score_mean": sum(scores) / len(scores),
        "class1_manipulated_score_min": min(scores),
        "class1_manipulated_score_max": max(scores),
        "frames": frames,
        "score_semantics": "Uncalibrated class-1 softmax score; not a probability of authenticity.",
        "elapsed_seconds": None,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        code = main()
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }, separators=(",", ":")))
        code = 1
    raise SystemExit(code)

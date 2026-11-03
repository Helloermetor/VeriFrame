
from __future__ import annotations
import os
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROOT = BACKEND_ROOT / "model_sources" / "fakestormer"
# Ensure official FakeSTormer modules resolve correctly.
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
CONFIG_PATH = (
    ROOT
    / "configs"
    / "temporal"
    / "FakeSFormer_base_c23_veriframe.yaml"
)

CHECKPOINT_PATH = (
    BACKEND_ROOT
    / "model_assets" / "FakeSTormer"
    / "TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth"
)

IMAGE_SIZE = (224, 224)
NUM_FRAMES = 4
CROP_BORDER = 15

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def extract_video_frames(video_path: Path) -> tuple[list[Image.Image], float, float, list[int]]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Video contains no readable frames.")

    if fps <= 0:
        fps = 0.0

    indices = np.linspace(
        0,
        max(0, frame_count - 1),
        NUM_FRAMES,
        dtype=int,
    )

    frames = []

    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()

            if not success:
                raise RuntimeError(
                    f"Could not read video frame at index {index}."
                )

            # OpenCV BGR -> RGB, matching PIL's RGB input.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            height, width = frame.shape[:2]

            if height <= 2 * CROP_BORDER or width <= 2 * CROP_BORDER:
                raise RuntimeError(
                    "Video frame is too small for the official 15-pixel crop."
                )

            frame = frame[
                CROP_BORDER:height - CROP_BORDER,
                CROP_BORDER:width - CROP_BORDER,
            ]

            frames.append(Image.fromarray(frame))

    finally:
        cap.release()

    duration = frame_count / fps if fps > 0 else 0.0

    return frames, fps, duration, [int(index) for index in indices]


def prepare_input(frames, transform, device):
    tensors = []

    for frame in frames:
        # Match official scripts/test.py preprocessing.
        resized = frame.resize(IMAGE_SIZE)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0

        tensor = transform(image_array).to(torch.float32)
        tensors.append(tensor)

    # [T, C, H, W] -> [1, C, T, H, W]
    video_tensor = torch.stack(tensors, dim=0)
    video_tensor = video_tensor.transpose(0, 1).unsqueeze(0)

    return video_tensor.to(device, non_blocking=True)


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

def run_inference(video_path: Path) -> dict:

    started = time.time()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. FakeSTormer requires GPU inference.")

    # Import official FakeSTormer components.
    from configs.get_config import load_config
    from models import MODELS, build_model, load_pretrained
    from package_utils.transform import final_transform

    torch.manual_seed(1024)
    np.random.seed(1024)
    torch.cuda.manual_seed_all(1024)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    cfg = load_config(str(CONFIG_PATH))

    if cfg.TEST.subtask != "test_vid":
        raise RuntimeError(
            "FakeSTormer configuration must use TEST.subtask='test_vid'."
        )

    if int(cfg.DATASET.DATA.SAMPLES_PER_VIDEO.NUM_FRAMES) != NUM_FRAMES:
        raise RuntimeError(
            "Configuration frame count does not match the runner."
        )

    if tuple(cfg.DATASET.IMAGE_SIZE) != IMAGE_SIZE:
        raise RuntimeError(
            "Configuration image size does not match the runner."
        )

    # Use absolute checkpoint path.
    cfg.TEST.pretrained = str(CHECKPOINT_PATH)

    frames, fps, duration, sampled_frame_indices = extract_video_frames(video_path)

    transform = final_transform(cfg.DATASET)
    video_tensor = prepare_input(frames, transform, device)

    # Official model construction and strict checkpoint loading.
    model = build_model(cfg.MODEL, MODELS).to(torch.float32)

    model = load_pretrained(model, str(CHECKPOINT_PATH))

    model = nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()

    # Sequential GPU inference.
    with torch.inference_mode():
        inference_started = time.time()
        outputs = model(video_tensor)
        inference_seconds = time.time() - inference_started

        temp_loc = outputs[0]["temp_loc"].sigmoid()
        cls_output = outputs[0]["cls"].sigmoid()

    temporal_values = (
        temp_loc.detach().float().cpu().numpy()[0].reshape(-1).tolist()
    )

    classification_values = (
        cls_output.detach().float().cpu().numpy()[0].reshape(-1).tolist()
    )

    if len(temporal_values) != NUM_FRAMES:
        raise RuntimeError(
            f"Expected {NUM_FRAMES} temporal values, "
            f"received {len(temporal_values)}."
        )

    threshold = float(cfg.TEST.threshold)
    final_temporal_value = float(temporal_values[-1])

    # Preserve the official FakeSTormer decision rule.
    official_label = (
        "Fake"
        if final_temporal_value > threshold
        else "Real"
    )

    checkpoint_hash = checkpoint_sha256(CHECKPOINT_PATH)

    result = {
        "status": "ok",
        "model_id": "fakestormer",
        "model_name": "FakeSTormer C23 ViT-Base",
        "model_role": "Challenger 2",
        "device": "cuda",
        "gpu_name": gpu_name,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(CONFIG_PATH),
        "sampled_frame_count": len(frames),
        "sampled_frame_indices": sampled_frame_indices,
        "source_fps": fps,
        "duration_seconds": duration,
        "image_size": list(IMAGE_SIZE),
        "crop_border": CROP_BORDER,
        "temporal_localization_values": temporal_values,
        "temporal_localization_final": final_temporal_value,
        "classification_head_values": classification_values,
        "threshold": threshold,
        "official_label": official_label,
        "score_semantics": (
            "Uncalibrated sigmoid temporal-localization outputs. "
            "Not calibrated probabilities. Official label follows "
            "the FakeSTormer test_vid threshold rule."
        ),
        "inference_seconds": inference_seconds,
        "elapsed_seconds": time.time() - started,
    }

    # Release GPU memory before subprocess exits.
    del model
    del video_tensor
    del outputs

    torch.cuda.empty_cache()

    return result


# --------------------------------------------------
# COMMAND LINE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="VeriFrame FakeSTormer video inference"
    )

    parser.add_argument("video", type=str, help="Input video path")

    args = parser.parse_args()

    try:
        result = run_inference(Path(args.video).resolve())
        print(json.dumps(result, allow_nan=False))

        return 0

    except Exception as exc:
        error_result = {
            "status": "error",
            "model_id": "fakestormer",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

        print(json.dumps(error_result, allow_nan=False))
        traceback.print_exc(file=sys.stderr)

        return 1


if __name__ == "__main__":
    raise SystemExit(main())


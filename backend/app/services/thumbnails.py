import os
import shutil
import subprocess
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import Response

FFMPEG_PATH = (
    os.getenv("VERIFRAME_FFMPEG_PATH")
    or shutil.which("ffmpeg")
)


def extract_thumbnail(video_path: str, timestamp: float) -> bytes:
    video = Path(video_path)

    if not video.is_file():
        raise HTTPException(404, "Original video file not found.")

    if timestamp < 0:
        raise HTTPException(400, "Invalid frame timestamp.")

    if not FFMPEG_PATH or not Path(FFMPEG_PATH).is_file():
        raise HTTPException(
            503,
            "FFmpeg is unavailable. Install FFmpeg or configure VERIFRAME_FFMPEG_PATH.",
        )

    command = [
        str(FFMPEG_PATH),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-ss",
        str(timestamp),
        "-frames:v",
        "1",
        "-vf",
        "scale=480:-2,format=yuv420p",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Thumbnail extraction timed out.")

    if result.returncode != 0 or not result.stdout:
        raise HTTPException(422, "Could not extract this video frame.")

    return result.stdout

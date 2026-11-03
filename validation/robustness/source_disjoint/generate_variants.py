from pathlib import Path
import csv
import subprocess
import shutil
import json
import os

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "validation" / "robustness" / "source_disjoint"
MANIFEST = WORK / "source_disjoint_20_pairs.csv"
OUTPUT = WORK / "videos"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Locate FFmpeg and FFprobe using optional overrides or PATH.
ffmpeg_override = os.getenv("VERIFRAME_FFMPEG_PATH")
ffprobe_override = os.getenv("VERIFRAME_FFPROBE_PATH")
ffmpeg = ffmpeg_override or shutil.which("ffmpeg")
ffprobe = ffprobe_override or shutil.which("ffprobe")

if not ffprobe and ffmpeg:
    candidate = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if candidate.is_file():
        ffprobe = str(candidate)

if not ffmpeg or not ffprobe:
    raise RuntimeError("FFmpeg or FFprobe not found.")

with MANIFEST.open("r", newline="", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))

if len(pairs) != 20:
    raise RuntimeError(f"Expected 20 pairs, found {len(pairs)}")

jobs = []

for pair in pairs:
    sid = pair["Source_ID"]

    for label, column in [
        ("original", "Original_Path"),
        ("fake", "Fake_Path")
    ]:
        source = Path(pair[column])

        if not source.is_file():
            raise FileNotFoundError(source)

        prefix = f"{label}_{sid}_{source.stem}"

        jobs.append({
            "source_id": sid,
            "class": label,
            "source": source,
            "transformation": "CRF32",
            "output": OUTPUT / f"{prefix}_crf32.mp4"
        })

        jobs.append({
            "source_id": sid,
            "class": label,
            "source": source,
            "transformation": "240p",
            "output": OUTPUT / f"{prefix}_240p.mp4"
        })

print("=" * 65)
print("SOURCE-DISJOINT ROBUSTNESS VIDEO GENERATION")
print("=" * 65)
print(f"Source pairs: {len(pairs)}")
print(f"Videos to transform: {len(jobs)}")
print(f"Output directory: {OUTPUT}")

results = []

for index, job in enumerate(jobs, 1):
    source = job["source"]
    output = job["output"]
    transformation = job["transformation"]

    if output.exists():
        raise RuntimeError(
            f"Output already exists; refusing to overwrite: {output}"
        )

    if transformation == "CRF32":
        video_args = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "32"
        ]
    else:
        video_args = [
            "-vf", "scale=-2:240",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23"
        ]

    command = [
        ffmpeg, "-y", "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a?",
        *video_args,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output)
    ]

    print(
        f"[{index}/{len(jobs)}] "
        f"{job['class']} | {source.name} | {transformation}"
    )

    process = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if process.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg failed for {source.name}:\n{process.stderr[-3000:]}"
        )

    results.append({
        "Source_ID": job["source_id"],
        "Class": job["class"],
        "Original_File": str(source),
        "Transformation": transformation,
        "Output_File": str(output),
        "Size_Bytes": output.stat().st_size
    })

    print(f"    Created: {output.name}")

# Save transformation manifest
manifest_out = WORK / "transformed_videos_manifest.csv"

with manifest_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Verify every output with FFprobe
properties = []

for index, item in enumerate(results, 1):
    command = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,bit_rate",
        "-show_entries", "format=duration,bit_rate",
        "-of", "json",
        item["Output_File"]
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"FFprobe failed: {item['Output_File']}"
        )

    data = json.loads(process.stdout)
    streams = data.get("streams", [])

    if not streams:
        raise RuntimeError(
            f"No video stream found: {item['Output_File']}"
        )

    stream = streams[0]
    fmt = data.get("format", {})

    width = int(stream["width"])
    height = int(stream["height"])

    if stream.get("codec_name") != "h264":
        raise RuntimeError(f"Unexpected codec: {item['Output_File']}")

    if item["Transformation"] == "240p" and height != 240:
        raise RuntimeError(
            f"240p verification failed: {item['Output_File']} ({height}p)"
        )

    properties.append({
        **item,
        "Codec": stream.get("codec_name"),
        "Width": width,
        "Height": height,
        "Duration_Seconds": fmt.get("duration"),
        "Bitrate": fmt.get("bit_rate")
    })

    if index % 10 == 0:
        print(f"Verified {index}/{len(results)} outputs")

properties_file = WORK / "transformed_video_properties.csv"

with properties_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=properties[0].keys())
    writer.writeheader()
    writer.writerows(properties)

print("\n" + "=" * 65)
print("GENERATION AND VERIFICATION COMPLETE")
print("=" * 65)
print(f"Total transformed videos: {len(results)}")
print(f"CRF32 videos: {sum(x['Transformation'] == 'CRF32' for x in results)}")
print(f"240p videos: {sum(x['Transformation'] == '240p' for x in results)}")
print(f"Output directory: {OUTPUT}")
print(f"Manifest: {manifest_out}")
print(f"Properties: {properties_file}")
print("All generated files passed FFprobe verification.")

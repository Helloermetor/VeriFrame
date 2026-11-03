import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from .thumbnails import extract_thumbnail


def _json_bytes(data):
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_evidence_package(record, pdf_bytes):
    analysis_id = record.get("analysis_id", "unknown")
    result = record.get("result") or {}

    metadata = result.get("metadata") or {}
    evidence = result.get("evidence") or {}
    stability = result.get("stability") or {}
    frames = result.get("frames") or []

    files = {}

    # Existing forensic PDF report
    files["report.pdf"] = pdf_bytes

    # Frame-level evidence CSV
    csv_buffer = io.StringIO()
    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "class1_manipulated_score",
        "threshold_model_output",
    ]

    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()

    valid_frames = []

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        row = {key: frame.get(key) for key in fieldnames}
        writer.writerow(row)
        valid_frames.append(frame)

    files["frame_scores.csv"] = csv_buffer.getvalue().encode("utf-8-sig")

    # Top five highest-scoring sampled frames
    top_frames = sorted(
        [
            frame
            for frame in valid_frames
            if isinstance(frame.get("class1_manipulated_score"), (int, float))
        ],
        key=lambda frame: frame["class1_manipulated_score"],
        reverse=True,
    )[:5]

    files["top_frames.json"] = _json_bytes(top_frames)
    # Include actual JPEG thumbnails for the top five frames
    video_path = record.get("video_path")

    if video_path:
        for frame in top_frames:
            timestamp = frame.get("timestamp_seconds")

            if timestamp is None:
                continue

            try:
                image_bytes = extract_thumbnail(
                    video_path,
                    float(timestamp),
                )

                frame_index = frame.get("frame_index", "unknown")
                filename = f"top_frames/frame_{frame_index}.jpg"

                files[filename] = image_bytes

            except Exception:
                # Preserve the rest of the package if one thumbnail fails.
                continue
    # Stability test results
    files["stability_results.json"] = _json_bytes(stability)

    # Analysis metadata and evidence
    files["analysis_metadata.json"] = _json_bytes(
        {
            "analysis_id": analysis_id,
            "created_at": record.get("created_at"),
            "status": record.get("status"),
            "outcome": result.get("outcome", "INCONCLUSIVE"),
            "outcome_note": result.get("outcome_note"),
            "metadata": metadata,
            "evidence": evidence,
            "committee": result.get("committee"),
        }
    )

    # Manifest with hashes of every included file
    manifest = {
        "package_name": "VeriFrame Forensic Evidence Package",
        "package_version": "1.0",
        "analysis_id": analysis_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": result.get("outcome", "INCONCLUSIVE"),
        "score_interpretation": (
            "Xception class-1 scores are uncalibrated model outputs, "
            "not probabilities or proof of manipulation."
        ),
        "files": {
            name: {
                "size_bytes": len(content),
                "sha256": _sha256(content),
            }
            for name, content in files.items()
        },
    }

    files["manifest.json"] = _json_bytes(manifest)

    # Build ZIP in memory
    output = io.BytesIO()

    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    return output.getvalue()

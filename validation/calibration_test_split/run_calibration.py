
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_DIR = ROOT / "validation" / "calibration_test_split"
MANIFEST = VALIDATION_DIR / "calibration_pairs.csv"

OUTPUT_DIR = (
    VALIDATION_DIR / "inference_results" / "calibration"
)

JSON_DIR = OUTPUT_DIR / "individual_results"

RESULTS_CSV = OUTPUT_DIR / "calibration_results.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"

# Import the existing VeriFrame inference service.
sys.path.insert(0, str(ROOT))

from backend.app.services.inference import run_committee


# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_manifest() -> list[dict[str, Any]]:

    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Calibration manifest not found: {MANIFEST}"
        )

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        rows = list(csv.DictReader(handle))

    videos = []

    for row in rows:

        if row.get("Split") != "calibration":
            raise ValueError(
                f"Unexpected split: {row.get('Split')}"
            )

        videos.append({
            "identity_group": row["Identity_Group"],
            "source_id": row["Source_ID"],
            "video_path": row["Original_Path"],
            "label": "original",
            "label_numeric": 0,
        })

        videos.append({
            "identity_group": row["Identity_Group"],
            "source_id": row["Source_ID"],
            "video_path": row["Fake_Path"],
            "label": "manipulated",
            "label_numeric": 1,
        })

    return videos


def output_name(video: dict[str, Any]) -> str:

    path = Path(video["video_path"])

    return (
        f"{video['identity_group']}__"
        f"{video['source_id']}__"
        f"{video['label']}__"
        f"{path.stem}.json"
    )


# ------------------------------------------------------------
# EXTRACT DETECTOR RESULTS
# ------------------------------------------------------------

def extract_result(
    metadata: dict[str, Any],
    committee: dict[str, Any],
) -> dict[str, Any]:

    gend = committee.get("gend") or {}
    xception = committee.get("xception") or {}
    effort = committee.get("effort") or {}
    fakestormer = committee.get("fakestormer") or {}

    return {
        **metadata,

        "inference_status": committee.get(
            "inference_status"
        ),

        # GenD
        "gend_status": gend.get("status", "failed"),
        "gend_score": gend.get(
            "class1_manipulated_score_mean"
        ),

        # Xception
        "xception_status": xception.get("status", "failed"),
        "xception_score": xception.get(
            "class1_manipulated_score_mean"
        ),

        # Effort
        "effort_status": effort.get("status", "failed"),
        "effort_score": effort.get(
            "class1_manipulated_score_mean"
        ),

        # FakeSTormer - kept separate
        "fakestormer_status": fakestormer.get(
            "status", "failed"
        ),

        "fakestormer_temporal_final": fakestormer.get(
            "temporal_localization_final"
        ),

        "fakestormer_official_label": fakestormer.get(
            "official_label"
        ),

        "fakestormer_threshold": fakestormer.get(
            "threshold"
        ),

        "model_errors": committee.get("model_errors"),
    }


# ------------------------------------------------------------
# SAVE RESULTS AND FAILURES
# ------------------------------------------------------------

def save_csvs() -> None:

    result_rows = []
    failure_rows = []

    for json_path in sorted(JSON_DIR.glob("*.json")):

        try:
            with json_path.open(
                "r",
                encoding="utf-8"
            ) as handle:
                record = json.load(handle)

        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read {json_path.name}: {exc}")
            continue

        result_rows.append(record["summary"])

        metadata = record["metadata"]
        committee = record.get("committee", {})

        errors = committee.get("model_errors") or {}

        for model, error in errors.items():

            if error:
                failure_rows.append({
                    "video_path": metadata["video_path"],
                    "identity_group": metadata["identity_group"],
                    "source_id": metadata["source_id"],
                    "label": metadata["label"],
                    "model": model,
                    "error": error,
                })

        if committee.get("inference_status") == "failed":

            failure_rows.append({
                "video_path": metadata["video_path"],
                "identity_group": metadata["identity_group"],
                "source_id": metadata["source_id"],
                "label": metadata["label"],
                "model": "committee",
                "error": committee.get(
                    "reason",
                    "All detectors failed."
                ),
            })

        if record.get("runner_error"):

            failure_rows.append({
                "video_path": metadata["video_path"],
                "identity_group": metadata["identity_group"],
                "source_id": metadata["source_id"],
                "label": metadata["label"],
                "model": "runner",
                "error": record["runner_error"].get("message"),
            })

    # Results CSV
    if result_rows:

        with RESULTS_CSV.open(
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(result_rows[0].keys()),
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(result_rows)

    # Failures CSV
    failure_fields = [
        "video_path",
        "identity_group",
        "source_id",
        "label",
        "model",
        "error",
    ]

    with FAILURES_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=failure_fields
        )

        writer.writeheader()
        writer.writerows(failure_rows)


# ------------------------------------------------------------
# MAIN CALIBRATION RUNNER
# ------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description="VeriFrame calibration-only inference runner."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum videos to attempt in this run."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos with successful four-detector results."
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    JSON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = read_manifest()

    print("=" * 65)
    print("VERIFRAME CALIBRATION INFERENCE")
    print("=" * 65)

    print(f"Project root: {ROOT}")
    print(f"Manifest: {MANIFEST}")
    print(f"Videos in manifest: {len(videos)}")
    print(f"Output directory: {OUTPUT_DIR}")

    print(
        "Independent test manifest will NOT be accessed."
    )

    print("=" * 65)

    attempted = 0
    successful = 0
    failed = 0

    for index, video in enumerate(videos, start=1):

        video_path = Path(video["video_path"])
        json_path = JSON_DIR / output_name(video)

        # Resume only fully completed committee results.
        if args.resume and json_path.is_file():

            try:
                existing = json.loads(
                    json_path.read_text(encoding="utf-8")
                )

                if (
                    existing.get("committee", {}).get(
                        "inference_status"
                    ) == "completed"
                ):
                    print(
                        f"[{index}/{len(videos)}] SKIP: "
                        f"{video_path.name}"
                    )
                    continue

            except (OSError, json.JSONDecodeError):
                pass

        if args.limit is not None and attempted >= args.limit:
            break

        attempted += 1

        print()
        print("-" * 65)
        print(f"[{index}/{len(videos)}] {video_path.name}")
        print(f"Identity group: {video['identity_group']}")
        print(f"Source ID: {video['source_id']}")
        print(f"Label: {video['label']}")
        print("-" * 65)

        started = time.time()

        metadata = {
            **video,
            "video_path": str(video_path.resolve()),
            "split": "calibration",
            "processed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        try:

            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video not found: {video_path}"
                )

            metadata["video_sha256"] = sha256_file(
                video_path
            )

            # Run all four existing detectors sequentially.
            committee = run_committee(
                str(video_path.resolve())
            )

            summary = extract_result(
                metadata,
                committee
            )

            record = {
                "metadata": metadata,
                "summary": summary,
                "committee": committee,
                "elapsed_seconds": round(
                    time.time() - started,
                    3
                ),
            }

            with json_path.open(
                "w",
                encoding="utf-8"
            ) as handle:

                json.dump(
                    record,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

            if committee.get("inference_status") == "completed":
                successful += 1
            else:
                failed += 1

            print(
                f"Committee: {committee.get('inference_status')}"
            )

            print("GenD:", summary["gend_score"])
            print("Xception:", summary["xception_score"])
            print("Effort:", summary["effort_score"])

            print(
                "FakeSTormer temporal:",
                summary["fakestormer_temporal_final"]
            )

            print(
                f"Elapsed: {record['elapsed_seconds']} seconds"
            )

            print(f"Saved: {json_path}")

        except Exception as exc:

            failed += 1

            error_record = {
                "metadata": metadata,
                "summary": {
                    **metadata,
                    "inference_status": "runner_error",
                },
                "committee": {
                    "inference_status": "failed",
                    "reason": str(exc),
                    "model_errors": {},
                },
                "runner_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "elapsed_seconds": round(
                    time.time() - started,
                    3
                ),
            }

            with json_path.open(
                "w",
                encoding="utf-8"
            ) as handle:

                json.dump(
                    error_record,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"ERROR: {type(exc).__name__}: {exc}"
            )

        # Keep CSVs updated after every video.
        save_csvs()

    save_csvs()

    print()
    print("=" * 65)
    print("CALIBRATION INFERENCE RUN FINISHED")
    print("=" * 65)

    print(f"Attempted this run: {attempted}")
    print(f"Fully successful: {successful}")
    print(f"Failed or partial: {failed}")

    print(f"JSON directory: {JSON_DIR}")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Failures CSV: {FAILURES_CSV}")

    print("=" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

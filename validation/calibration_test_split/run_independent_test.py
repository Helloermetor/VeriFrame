from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

MANIFEST = VALIDATION_DIR / "independent_test_pairs.csv"
CALIBRATION_MANIFEST = VALIDATION_DIR / "calibration_pairs.csv"
CALIBRATION_RESULTS_CSV = (
    VALIDATION_DIR / "inference_results" / "calibration" / "calibration_results.csv"
)
PARAMETERS_FILE = VALIDATION_DIR / "calibration_parameters.json"

OUTPUT_DIR = VALIDATION_DIR / "inference_results" / "independent_test"

JSON_DIR = OUTPUT_DIR / "individual_results"

RESULTS_CSV = OUTPUT_DIR / "independent_test_results.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"

sys.path.insert(0, str(ROOT))

from backend.app.services.inference import run_committee

# ------------------------------------------------------------
# FROZEN CALIBRATION
# ------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_calibration_parameters() -> dict[str, Any]:
    if not PARAMETERS_FILE.is_file():
        raise FileNotFoundError(PARAMETERS_FILE)

    parameters = json.loads(PARAMETERS_FILE.read_text(encoding="utf-8"))

    if parameters.get("independent_test_accessed") is not False:
        raise ValueError("Calibration artifact does not confirm an untouched test set.")

    if parameters.get("production_pipeline_modified") is not False:
        raise ValueError("Unexpected production pipeline modification flag.")

    # Preserve the original frozen hash; do not replace it.
    expected_hash = parameters["source_csv_sha256"]
    actual_hash = sha256_file(CALIBRATION_RESULTS_CSV)

    # Verify calibration manifest against the original inference records.
    with CALIBRATION_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    results_dir = (
        VALIDATION_DIR / "inference_results" / "calibration" / "individual_results"
    )

    expected = {}

    for row in rows:
        for field, label in (
            ("Original_Path", "original"),
            ("Fake_Path", "manipulated"),
        ):
            video_path = Path(row[field]).resolve()

            key = (
                row["Identity_Group"],
                str(row["Source_ID"]),
                label,
                video_path.stem,
            )

            if key in expected:
                raise ValueError(f"Duplicate calibration manifest record: {key}")

            expected[key] = str(video_path).lower()

    actual = {}

    for json_path in results_dir.glob("*.json"):
        record = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = record["metadata"]

        video_path = Path(metadata["video_path"]).resolve()

        key = (
            str(metadata["identity_group"]),
            str(metadata["source_id"]),
            str(metadata["label"]),
            video_path.stem,
        )

        if key in actual:
            raise ValueError(f"Duplicate calibration inference record: {key}")

        actual[key] = str(video_path).lower()

    missing = set(expected) - set(actual)
    unexpected = set(actual) - set(expected)

    path_mismatches = [
        key for key in set(expected) & set(actual) if expected[key] != actual[key]
    ]

    if (
        len(rows) != 24
        or len(expected) != 48
        or len(actual) != 48
        or missing
        or unexpected
        or path_mismatches
    ):
        raise ValueError(
            "Calibration semantic integrity audit failed. "
            f"Rows={len(rows)}, expected={len(expected)}, "
            f"actual={len(actual)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, "
            f"path_mismatches={len(path_mismatches)}."
        )

    calibration_groups = {row["Identity_Group"] for row in rows}

    if len(calibration_groups) != 12:
        raise ValueError("Expected exactly 12 calibration identity groups.")

    models = parameters["calibrated_models"]

    for model in ("gend", "xception"):
        if model not in models:
            raise ValueError(f"Missing frozen calibration: {model}")

    # Attach audit information in memory only.
    # The frozen JSON artifact remains unchanged.
    parameters["_manifest_integrity_audit"] = {
        "verification": "semantic_match",
        "frozen_sha256": expected_hash,
        "current_sha256": actual_hash,
        "byte_hash_match": actual_hash == expected_hash,
        "manifest_rows": len(rows),
        "calibration_videos": len(expected),
        "inference_records": len(actual),
        "identity_groups": len(calibration_groups),
        "missing_records": 0,
        "unexpected_records": 0,
        "path_mismatches": 0,
        "independent_test_accessed": False,
    }

    return parameters


def logit(score: float) -> float:

    epsilon = 1e-7
    score = min(max(float(score), epsilon), 1 - epsilon)

    return math.log(score / (1 - score))


def sigmoid(value: float) -> float:

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def apply_platt(
    score: Any,
    parameters: dict[str, Any],
    model: str,
) -> float | None:

    if score is None:
        return None

    try:
        raw_score = float(score)

        if not math.isfinite(raw_score):
            return None

        config = parameters["calibrated_models"][model]

        transformed = logit(raw_score)

        return sigmoid(config["coefficient"] * transformed + config["intercept"])

    except (TypeError, ValueError, KeyError):
        return None


# ------------------------------------------------------------
# TEST MANIFEST
# ------------------------------------------------------------


def read_manifest() -> list[dict[str, Any]]:

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        rows = list(csv.DictReader(handle))

    videos = []

    for row in rows:

        if row.get("Split") != "independent_test":
            raise ValueError(f"Unexpected split: {row.get('Split')}")

        for path_field, label, numeric_label in (
            ("Original_Path", "original", 0),
            ("Fake_Path", "manipulated", 1),
        ):

            video_path = Path(row[path_field])

            if not video_path.is_file():
                raise FileNotFoundError(video_path)

            videos.append(
                {
                    "identity_group": row["Identity_Group"],
                    "source_id": row["Source_ID"],
                    "video_path": str(video_path.resolve()),
                    "label": label,
                    "label_numeric": numeric_label,
                    "split": "independent_test",
                }
            )

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
# RESULT EXTRACTION
# ------------------------------------------------------------


def extract_result(
    metadata: dict[str, Any],
    committee: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:

    gend = committee.get("gend") or {}
    xception = committee.get("xception") or {}
    effort = committee.get("effort") or {}
    fakestormer = committee.get("fakestormer") or {}

    gend_raw = gend.get("class1_manipulated_score_mean")
    xception_raw = xception.get("class1_manipulated_score_mean")
    effort_raw = effort.get("class1_manipulated_score_mean")

    return {
        **metadata,
        "inference_status": committee.get("inference_status"),
        # GenD
        "gend_status": gend.get("status", "failed"),
        "gend_score_raw": gend_raw,
        "gend_score_calibrated": apply_platt(gend_raw, parameters, "gend"),
        # Xception
        "xception_status": xception.get("status", "failed"),
        "xception_score_raw": xception_raw,
        "xception_score_calibrated": apply_platt(xception_raw, parameters, "xception"),
        # Effort remains uncalibrated
        "effort_status": effort.get("status", "failed"),
        "effort_score_raw": effort_raw,
        "effort_calibration": "none",
        # FakeSTormer remains separate
        "fakestormer_status": fakestormer.get("status", "failed"),
        "fakestormer_temporal_final": fakestormer.get("temporal_localization_final"),
        "fakestormer_official_label": fakestormer.get("official_label"),
        "fakestormer_threshold": fakestormer.get("threshold"),
        "model_errors": committee.get("model_errors"),
    }


# ------------------------------------------------------------
# CSV EXPORT
# ------------------------------------------------------------


def save_csvs() -> None:

    result_rows = []
    failure_rows = []

    for json_path in sorted(JSON_DIR.glob("*.json")):

        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read {json_path.name}: {exc}")
            continue

        result_rows.append(record["summary"])

        metadata = record["metadata"]
        committee = record.get("committee", {})

        errors = committee.get("model_errors") or {}

        for model, error in errors.items():
            if error:
                failure_rows.append(
                    {
                        "video_path": metadata["video_path"],
                        "identity_group": metadata["identity_group"],
                        "source_id": metadata["source_id"],
                        "label": metadata["label"],
                        "model": model,
                        "error": error,
                    }
                )

        if record.get("runner_error"):
            failure_rows.append(
                {
                    "video_path": metadata["video_path"],
                    "identity_group": metadata["identity_group"],
                    "source_id": metadata["source_id"],
                    "label": metadata["label"],
                    "model": "runner",
                    "error": record["runner_error"].get("message"),
                }
            )

    if result_rows:
        with RESULTS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(result_rows[0].keys()),
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(result_rows)

    failure_fields = [
        "video_path",
        "identity_group",
        "source_id",
        "label",
        "model",
        "error",
    ]

    with FAILURES_CSV.open("w", encoding="utf-8-sig", newline="") as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=failure_fields,
        )

        writer.writeheader()
        writer.writerows(failure_rows)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser(
        description="VeriFrame independent test inference runner."
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    print("=" * 65)
    print("VERIFRAME INDEPENDENT TEST")
    print("=" * 65)

    # Validate frozen calibration before accessing test inference.
    parameters = load_calibration_parameters()

    videos = read_manifest()

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    print("Test manifest:", MANIFEST)
    print("Videos:", len(videos))
    print("Identity groups:", len({video["identity_group"] for video in videos}))
    print("Output:", OUTPUT_DIR)

    print("\nFrozen calibration:")
    print("GenD:", parameters["calibrated_models"]["gend"])
    print("Xception:", parameters["calibrated_models"]["xception"])
    print("Effort: uncalibrated")
    print("FakeSTormer: separate temporal evidence")

    print("\nProduction pipeline will NOT be modified.")
    print("=" * 65)

    attempted = 0
    successful = 0
    failed = 0

    for index, video in enumerate(videos, start=1):

        video_path = Path(video["video_path"])
        json_path = JSON_DIR / output_name(video)

        if args.resume and json_path.is_file():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))

                if existing.get("committee", {}).get(
                    "inference_status"
                ) == "completed" and not existing.get("runner_error"):
                    print(f"[{index}/{len(videos)}] SKIP: " f"{video_path.name}")
                    continue

            except (OSError, json.JSONDecodeError):
                pass

        if args.limit is not None and attempted >= args.limit:
            break

        attempted += 1

        print()
        print("-" * 65)
        print(f"[{index}/{len(videos)}] {video_path.name}")
        print("Identity group:", video["identity_group"])
        print("Label:", video["label"])
        print("-" * 65)

        started = time.time()

        metadata = {
            **video,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            "video_sha256": sha256_file(video_path),
        }

        try:
            committee = run_committee(str(video_path))

            summary = extract_result(
                metadata,
                committee,
                parameters,
            )

            record = {
                "metadata": metadata,
                "summary": summary,
                "committee": committee,
                "elapsed_seconds": round(time.time() - started, 3),
            }

            with json_path.open("w", encoding="utf-8") as handle:
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

            print("Committee:", committee.get("inference_status"))
            print("GenD raw:", summary["gend_score_raw"])
            print("GenD calibrated:", summary["gend_score_calibrated"])
            print("Xception raw:", summary["xception_score_raw"])
            print(
                "Xception calibrated:",
                summary["xception_score_calibrated"],
            )
            print("Effort raw:", summary["effort_score_raw"])
            print(
                "FakeSTormer temporal:",
                summary["fakestormer_temporal_final"],
            )

            print("Elapsed:", record["elapsed_seconds"], "seconds")
            print("Saved:", json_path)

        except Exception as exc:

            failed += 1

            record = {
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
                "elapsed_seconds": round(time.time() - started, 3),
            }

            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    record,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            print("ERROR:", type(exc).__name__, exc)

        save_csvs()

    save_csvs()

    print()
    print("=" * 65)
    print("INDEPENDENT TEST INFERENCE FINISHED")
    print("=" * 65)
    print("Attempted:", attempted)
    print("Fully successful:", successful)
    print("Failed or partial:", failed)
    print("Results:", RESULTS_CSV)
    print("Failures:", FAILURES_CSV)
    print("JSON:", JSON_DIR)
    print("=" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv
import json
import time
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "validation" / "robustness" / "source_disjoint"

PAIR_MANIFEST = WORK / "source_disjoint_20_pairs.csv"
TRANSFORM_MANIFEST = WORK / "transformed_videos_manifest.csv"

OUT_DIR = WORK / "api_results"
RESULT_DIR = OUT_DIR / "individual_results"

STATE_FILE = OUT_DIR / "submission_state.json"
SUMMARY_FILE = OUT_DIR / "source_disjoint_results.csv"

BASE = "http://127.0.0.1:8000/api/v1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def api_get(url, attempts=8):
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = SESSION.get(url, timeout=(15, 180))
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            wait = min(attempt * 5, 30)
            print(f"GET retry {attempt}/{attempts}: {exc}")
            time.sleep(wait)

    raise RuntimeError(f"GET failed after retries: {last_error}")


def build_jobs():
    jobs = []

    with PAIR_MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))

    if len(pairs) != 20:
        raise RuntimeError(f"Expected 20 pairs, found {len(pairs)}")

    for pair in pairs:
        sid = pair["Source_ID"].strip()

        for label, column in [
            ("original", "Original_Path"),
            ("fake", "Fake_Path"),
        ]:
            path = Path(pair[column].strip())

            if not path.is_file():
                raise FileNotFoundError(path)

            jobs.append({
                "key": f"{label}_{sid}_baseline",
                "source_id": sid,
                "class": label,
                "transformation": "baseline",
                "path": str(path),
            })

    with TRANSFORM_MANIFEST.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        variants = list(csv.DictReader(f))

    if len(variants) != 80:
        raise RuntimeError(
            f"Expected 80 transformed videos, found {len(variants)}"
        )

    for item in variants:
        sid = item["Source_ID"].strip()
        label = item["Class"].strip().lower()
        transformation = item["Transformation"].strip()
        path = Path(item["Output_File"].strip())

        if label not in ("original", "fake"):
            raise RuntimeError(f"Unexpected class: {label}")

        if transformation not in ("CRF32", "240p"):
            raise RuntimeError(
                f"Unexpected transformation: {transformation}"
            )

        if not path.is_file():
            raise FileNotFoundError(path)

        jobs.append({
            "key": f"{label}_{sid}_{transformation.lower()}",
            "source_id": sid,
            "class": label,
            "transformation": transformation,
            "path": str(path),
        })

    keys = [job["key"] for job in jobs]

    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate job keys detected.")

    return jobs


def upload_video(job):
    path = Path(job["path"])

    print(f"Uploading: {path.name}")

    # Do not automatically retry POST uploads: if a connection fails
    # after the server accepts a request, retrying could create duplicates.
    with path.open("rb") as video:
        response = SESSION.post(
            f"{BASE}/analyses",
            files={"video": (path.name, video, "video/mp4")},
            data={"consent": "true"},
            timeout=(30, 600),
        )

    response.raise_for_status()
    data = response.json()

    if not data.get("analysis_id"):
        raise RuntimeError(f"No analysis ID returned: {data}")

    return data["analysis_id"]


def collect_evidence(analysis_id):
    try:
        return api_get(
            f"{BASE}/analyses/{analysis_id}/evidence"
        )
    except Exception as exc:
        print(f"Evidence warning for {analysis_id}: {exc}")
        return {"retrieval_error": str(exc)}


def save_completed(job, analysis_id, record):
    evidence = collect_evidence(analysis_id)

    payload = {
        **job,
        "analysis_id": analysis_id,
        "record": record,
        "evidence": evidence,
    }

    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", job["key"])
    save_json(RESULT_DIR / f"{safe_key}.json", payload)

    print(f"Saved result: {safe_key}.json")
    return payload


def wait_for_completion(job, analysis_id, state):
    start = time.time()
    timeout_seconds = 3600

    while True:
        record = api_get(
            f"{BASE}/analyses/{analysis_id}"
        )

        status = str(record.get("status", "")).lower()

        print(
            f"{job['key']}: status={status}, "
            f"stage={record.get('stage')}, "
            f"progress={record.get('progress')}"
        )

        if status == "completed":
            payload = save_completed(job, analysis_id, record)

            state[job["key"]]["status"] = "completed"
            save_json(STATE_FILE, state)

            return payload

        if status == "failed":
            state[job["key"]]["status"] = "failed"
            state[job["key"]]["error"] = record.get("error")
            save_json(STATE_FILE, state)

            raise RuntimeError(
                f"Analysis failed for {job['key']}: "
                f"{record.get('error')}"
            )

        if time.time() - start > timeout_seconds:
            state[job["key"]]["status"] = "timeout"
            save_json(STATE_FILE, state)

            raise TimeoutError(
                f"Timed out waiting for {job['key']}. "
                "Its analysis ID is saved; rerun to resume."
            )

        time.sleep(10)


def export_summary():
    rows = []

    for path in RESULT_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = payload.get("record", {})
            result = record.get("result") or {}
            committee = result.get("committee") or {}

            row = {
                "Source_ID": payload.get("source_id"),
                "Class": payload.get("class"),
                "Transformation": payload.get("transformation"),
                "Filename": Path(payload.get("path", "")).name,
                "Analysis_ID": payload.get("analysis_id"),
                "Status": record.get("status"),
                "Verdict": result.get("verdict"),
                "GenD": json.dumps(committee.get("gend")),
                "Xception": json.dumps(committee.get("xception")),
                "Effort": json.dumps(committee.get("effort")),
                "FakeSTormer": json.dumps(committee.get("fakestormer")),
                "Result_JSON": json.dumps(result),
                "Evidence_JSON": json.dumps(payload.get("evidence")),
            }

            rows.append(row)

        except Exception as exc:
            print(f"Could not read {path.name}: {exc}")

    if rows:
        fields = list(rows[0].keys())

        with SUMMARY_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported result rows: {len(rows)}")
    print(f"Summary: {SUMMARY_FILE}")


def main():
    health = api_get(f"{BASE}/health")
    print("API health:", health)

    if health.get("status") != "ok":
        raise RuntimeError("API health check failed.")

    jobs = build_jobs()
    state = load_state()

    print("\n" + "=" * 65)
    print("VERIFRAME SOURCE-DISJOINT VALIDATION")
    print("=" * 65)
    print(f"Total jobs: {len(jobs)}")
    print(f"Baseline videos: 40")
    print(f"CRF32 videos: 40")
    print(f"240p videos: 40")
    print(f"Output directory: {OUT_DIR}")
    print("\nStarting/resuming sequential analysis.")
    print("Overall verdict policy remains INCONCLUSIVE.")

    for index, job in enumerate(jobs, 1):
        key = job["key"]

        print("\n" + "-" * 65)
        print(f"VIDEO {index}/{len(jobs)}: {key}")
        print("-" * 65)

        existing = state.get(key, {})
        analysis_id = existing.get("analysis_id")

        if analysis_id:
            print(f"Resuming saved analysis ID: {analysis_id}")

        else:
            analysis_id = upload_video(job)

            state[key] = {
                "analysis_id": analysis_id,
                "status": "submitted",
                "source_id": job["source_id"],
                "class": job["class"],
                "transformation": job["transformation"],
                "path": job["path"],
            }

            # Persist the ID immediately after successful upload.
            save_json(STATE_FILE, state)
            print(f"Submitted: {analysis_id}")

        wait_for_completion(job, analysis_id, state)

        export_summary()

    print("\n" + "=" * 65)
    print("ALL 120 ANALYSES COMPLETED")
    print("=" * 65)

    export_summary()
    print("State file:", STATE_FILE)
    print("Individual results:", RESULT_DIR)


if __name__ == "__main__":
    main()

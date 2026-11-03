import csv
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
DATA_ROOT = Path(os.getenv("VERIFRAME_FACEFORENSICS_ROOT", str(ROOT / "datasets" / "FaceForensics++" / "data"))).expanduser().resolve()
CSV_FILE = VALIDATION / "matched_10_pairs.csv"
OUT_DIR = VALIDATION / "batch_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "http://127.0.0.1:8000/api/v1"
SESSION = requests.Session()

# Previously completed original video: Source ID 033
KNOWN_COMPLETED = {
    "033.mp4": "775a4b3e-7493-4f3d-88f3-8dd635131870"
}

def request_json(method, url, **kwargs):
    last_error = None
    for attempt in range(1, 9):
        try:
            response = SESSION.request(
                method, url, timeout=(15, 180), **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            wait = min(attempt * 5, 30)
            print(f"API retry {attempt}/8: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"API unavailable after retries: {last_error}")

def get_analysis(analysis_id):
    return request_json("GET", f"{BASE}/analyses/{analysis_id}")

def save_result(name, analysis_id, record):
    evidence = None
    try:
        evidence = request_json(
            "GET", f"{BASE}/analyses/{analysis_id}/evidence"
        )
    except Exception as exc:
        print(f"Evidence retrieval warning: {exc}")

    payload = {
        "filename": name,
        "analysis_id": analysis_id,
        "record": record,
        "evidence": evidence,
    }

    path = OUT_DIR / f"{analysis_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {path.name}")

    return payload

def wait_for_completion(name, analysis_id):
    while True:
        record = get_analysis(analysis_id)
        status = str(record.get("status", "")).lower()

        print(
            f"{name}: status={status}, "
            f"stage={record.get('stage')}, "
            f"progress={record.get('progress')}"
        )

        if status == "completed":
            return save_result(name, analysis_id, record)

        if status == "failed":
            save_result(name, analysis_id, record)
            raise RuntimeError(
                f"Analysis failed for {name}: {record.get('error')}"
            )

        time.sleep(10)

def upload_video(path):
    with path.open("rb") as video:
        response = SESSION.post(
            f"{BASE}/analyses",
            files={"video": (path.name, video, "video/mp4")},
            data={"consent": "true"},
            timeout=(30, 300),
        )
    response.raise_for_status()
    return response.json()["analysis_id"]

def process_video(relative_path, role):
    path = DATA_ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(path)

    name = path.name
    print(f"\n--- {role}: {name} ---")

    # Reuse the known completed original analysis.
    if name in KNOWN_COMPLETED:
        analysis_id = KNOWN_COMPLETED[name]
        record = get_analysis(analysis_id)
        if str(record.get("status", "")).lower() == "completed":
            print("Reusing previously completed analysis.")
            return save_result(name, analysis_id, record)

    # Resume from a saved result if available.
    for saved in OUT_DIR.glob("*.json"):
        try:
            data = json.loads(saved.read_text(encoding="utf-8"))
            if data.get("filename") == name:
                old_id = data.get("analysis_id")
                if old_id:
                    record = get_analysis(old_id)
                    if str(record.get("status", "")).lower() == "completed":
                        print("Reusing saved completed analysis.")
                        return save_result(name, old_id, record)
        except Exception:
            pass

    analysis_id = upload_video(path)
    print(f"Uploaded. Analysis ID: {analysis_id}")
    return wait_for_completion(name, analysis_id)

def main():
    health = request_json("GET", f"{BASE}/health")
    print("API health:", health)
    print("\nStarting/resuming matched validation.")
    print("Overall verdict remains INCONCLUSIVE.")

    with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))

    all_results = []

    for i, pair in enumerate(pairs, 1):
        source_id = str(pair["Source_ID"]).strip()
        original = pair["Original"].strip()
        fake = pair["Fake"].strip()

        print(f"\n{'='*55}")
        print(f"PAIR {i}/{len(pairs)} — Source ID {source_id}")
        print(f"{'='*55}")

        for role, relative in [
            ("ORIGINAL", original),
            ("FAKE", fake),
        ]:
            result = process_video(relative, role)
            all_results.append(result)

            (VALIDATION / "all_raw_results.json").write_text(
                json.dumps(all_results, indent=2),
                encoding="utf-8"
            )

    print("\nAll videos processed. Building result index.")

    rows = []
    for saved in OUT_DIR.glob("*.json"):
        try:
            data = json.loads(saved.read_text(encoding="utf-8"))
            record = data.get("record", {})
            result = record.get("result") or {}
            rows.append({
                "filename": data.get("filename"),
                "analysis_id": data.get("analysis_id"),
                "status": record.get("status"),
                "verdict": result.get("verdict"),
                "result_json": json.dumps(result),
                "evidence_json": json.dumps(data.get("evidence")),
            })
        except Exception as exc:
            print(f"Could not index {saved.name}: {exc}")

    if rows:
        with (VALIDATION / "matched_validation_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("\nVALIDATION PROCESS FINISHED.")
    print("Raw results:", OUT_DIR)
    print("CSV:", VALIDATION / "matched_validation_results.csv")
    print("Remember: scores are uncalibrated; verdict remains INCONCLUSIVE.")

if __name__ == "__main__":
    main()

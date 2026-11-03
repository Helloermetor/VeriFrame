import hashlib
import io
import json
import zipfile


def verify_evidence_package(zip_bytes: bytes) -> dict:
    result = {
        "status": "INCOMPLETE",
        "verified": False,
        "total_files": 0,
        "verified_files": 0,
        "missing_files": [],
        "modified_files": [],
        "errors": [],
        "analysis_id": None,
        "verdict": None,
    }

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            if archive.testzip() is not None:
                result["errors"].append("ZIP archive integrity check failed.")
                return result

            names = archive.namelist()

            if "manifest.json" not in names:
                result["errors"].append("manifest.json is missing.")
                return result

            manifest = json.loads(archive.read("manifest.json"))

            if not isinstance(manifest, dict):
                result["errors"].append("Invalid manifest format.")
                return result

            expected_files = manifest.get("files")

            if not isinstance(expected_files, dict) or not expected_files:
                result["errors"].append("Manifest does not contain a valid files list.")
                return result

            result["analysis_id"] = manifest.get("analysis_id")
            result["verdict"] = manifest.get("verdict")
            result["total_files"] = len(expected_files)

            for filename, expected in expected_files.items():
                if not isinstance(filename, str) or not isinstance(expected, dict):
                    result["errors"].append("Invalid manifest file entry.")
                    continue

                if filename not in names:
                    result["missing_files"].append(filename)
                    continue

                content = archive.read(filename)
                actual_hash = hashlib.sha256(content).hexdigest()
                actual_size = len(content)

                if actual_hash != expected.get("sha256") or actual_size != expected.get(
                    "size_bytes"
                ):
                    result["modified_files"].append(
                        {
                            "filename": filename,
                            "expected_sha256": expected.get("sha256"),
                            "actual_sha256": actual_hash,
                            "expected_size_bytes": expected.get("size_bytes"),
                            "actual_size_bytes": actual_size,
                        }
                    )
                else:
                    result["verified_files"] += 1

            if result["errors"] or result["missing_files"]:
                result["status"] = "INCOMPLETE"
            elif result["modified_files"]:
                result["status"] = "TAMPERED"
            else:
                result["status"] = "VERIFIED"

            result["verified"] = result["status"] == "VERIFIED"

    except zipfile.BadZipFile:
        result["errors"].append("Uploaded file is not a valid ZIP archive.")
    except (json.JSONDecodeError, UnicodeDecodeError):
        result["errors"].append("manifest.json is invalid or unreadable.")
    except Exception as exc:
        result["errors"].append(f"Verification failed: {str(exc)}")

    return result

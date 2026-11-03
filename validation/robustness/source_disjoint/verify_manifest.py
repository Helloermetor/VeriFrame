from pathlib import Path
import csv

manifest = Path(__file__).resolve().parent / "source_disjoint_20_pairs.csv"

with manifest.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

missing = []
source_ids = set()
duplicate_ids = []

for row in rows:
    sid = row["Source_ID"]

    if sid in source_ids:
        duplicate_ids.append(sid)
    source_ids.add(sid)

    for label in ("Original_Path", "Fake_Path"):
        path = Path(row[label])
        if not path.is_file():
            missing.append((sid, label, str(path)))

print("=" * 55)
print("SOURCE-DISJOINT MANIFEST VERIFICATION")
print("=" * 55)
print(f"Matched pairs: {len(rows)}")
print(f"Original videos: {len(rows)}")
print(f"Fake videos: {len(rows)}")
print(f"Unique source IDs: {len(source_ids)}")
print(f"Missing files: {len(missing)}")
print(f"Duplicate source IDs: {len(duplicate_ids)}")

if missing:
    print("\nMISSING FILES:")
    for item in missing:
        print(item)

if duplicate_ids:
    print("\nDUPLICATE SOURCE IDs:", duplicate_ids)

if len(rows) == 20 and not missing and not duplicate_ids:
    print("\nSUCCESS: All 40 video paths verified.")
else:
    print("\nVERIFICATION FAILED: Check the manifest.")

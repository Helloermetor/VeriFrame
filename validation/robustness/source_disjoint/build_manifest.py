from pathlib import Path
import os
import csv
import random

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("VERIFRAME_FACEFORENSICS_ROOT", str(ROOT / "datasets" / "FaceForensics++" / "data"))).expanduser().resolve()
original_dir = DATA_ROOT / "original_sequences" / "youtube" / "c23" / "videos"
fake_dir = DATA_ROOT / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
output_dir = Path(__file__).resolve().parent

excluded = {"033", "035", "036", "044", "046", "055", "062", "066", "097", "107"}

originals = {p.stem: p for p in original_dir.glob("*.mp4")}

fake_by_source = {}
for p in fake_dir.glob("*.mp4"):
    parts = p.stem.split("_")
    if len(parts) >= 2:
        fake_by_source.setdefault(parts[0], []).append(p)

candidates = []
for source_id, original_path in originals.items():
    if source_id in excluded:
        continue

    fakes = sorted(fake_by_source.get(source_id, []))
    if fakes:
        candidates.append((source_id, original_path, fakes[0]))

random.Random(2026).shuffle(candidates)
selected = candidates[:20]

if len(selected) < 20:
    raise RuntimeError(
        f"Only {len(selected)} eligible matched pairs found. "
        "Check dataset paths and available files."
    )

output_dir.mkdir(parents=True, exist_ok=True)
manifest = output_dir / "source_disjoint_20_pairs.csv"

with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Source_ID", "Original_Path", "Fake_Path"])
    for source_id, original_path, fake_path in selected:
        writer.writerow([source_id, str(original_path), str(fake_path)])

print(f"Selected pairs: {len(selected)}")
print(f"Excluded source IDs: {', '.join(sorted(excluded))}")
print(f"Manifest: {manifest}")
for source_id, original_path, fake_path in selected:
    print(f"{source_id}: {original_path.name} -> {fake_path.name}")

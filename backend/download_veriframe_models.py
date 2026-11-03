from pathlib import Path
import hashlib
import gdown
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "model_assets"

GEND_DIR = ASSETS / "GenD"
CLIP_DIR = GEND_DIR / "clip-vit-large-patch14"
EFFORT_DIR = ASSETS / "Effort"

GEND_MODEL_ID = "yermandy/GenD_CLIP_L_14"
GEND_REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"

EFFORT_DRIVE_ID = "1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI"

EXPECTED_HASHES = {
    "gend": "D76F0BDFD74A29FE1B1C1B84A80AC92486993E426878E8C7A3944281FBB96833",
    "clip": "A2BF730A0C7DEBF160F7A6B50B3AAF3703E7E88AC73DE7A314903141DB026DCB",
    "effort": "8D86711F098D16B49C048962FC3E16A906380F7BB17B8A0E89BD545B926943EE",
}

CLIP_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]

EFFORT_CHECKPOINT = EFFORT_DIR / "effort_clip_L14_trainOn_FaceForensic.pth"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify(path, expected_hash, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")

    actual = sha256(path)
    print(f"{label} SHA256: {actual}")

    if actual != expected_hash:
        raise RuntimeError(
            f"{label} SHA256 mismatch. Expected {expected_hash}, got {actual}"
        )

    print(f"{label}: verified")


def download_effort():
    EFFORT_DIR.mkdir(parents=True, exist_ok=True)

    if EFFORT_CHECKPOINT.exists():
        if sha256(EFFORT_CHECKPOINT) == EXPECTED_HASHES["effort"]:
            print("\nEffort checkpoint already verified; skipping download.")
            return
        print("\nExisting Effort checkpoint is not the expected file.")

    temporary = EFFORT_CHECKPOINT.with_name(
        EFFORT_CHECKPOINT.name + ".download"
    )

    if temporary.exists():
        temporary.unlink()

    print("\nDownloading official Effort FaceForensics++ checkpoint...")
    result = gdown.download(
        id=EFFORT_DRIVE_ID,
        output=str(temporary),
        quiet=False,
    )

    if not result or not temporary.is_file():
        raise RuntimeError("Effort checkpoint download failed.")

    verify(temporary, EXPECTED_HASHES["effort"], "Effort checkpoint")
    temporary.replace(EFFORT_CHECKPOINT)
    print("Effort checkpoint saved:", EFFORT_CHECKPOINT)


def main():
    GEND_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("VERIFRAME MODEL ASSET DOWNLOADER")
    print("=" * 65)

    print("\nDownloading/verifying GenD checkpoint...")
    snapshot_download(
        repo_id=GEND_MODEL_ID,
        revision=GEND_REVISION,
        local_dir=str(GEND_DIR),
        allow_patterns=["config.json", "model.safetensors"],
    )
    verify(GEND_DIR / "model.safetensors", EXPECTED_HASHES["gend"], "GenD")

    print("\nDownloading/verifying shared CLIP backbone...")
    snapshot_download(
        repo_id=CLIP_MODEL_ID,
        revision=CLIP_REVISION,
        local_dir=str(CLIP_DIR),
        allow_patterns=CLIP_FILES,
    )

    missing = [name for name in CLIP_FILES if not (CLIP_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CLIP files missing: {missing}")

    verify(CLIP_DIR / "model.safetensors", EXPECTED_HASHES["clip"], "CLIP")

    print("\nEffort uses this shared CLIP directory:")
    print(CLIP_DIR)

    download_effort()

    print("\n" + "=" * 65)
    print("GEN D, SHARED CLIP AND EFFORT ASSETS VERIFIED")
    print("=" * 65)
    print("FakeSTormer checkpoint must currently be downloaded separately.")
    print("See the official FakeSTormer README for its Dropbox download.")
    print("=" * 65)


if __name__ == "__main__":
    main()

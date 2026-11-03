from pathlib import Path
from huggingface_hub import snapshot_download

GEND_MODEL_ID = "yermandy/GenD_CLIP_L_14"
GEND_MODEL_REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_MODEL_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"

MODEL_DIR = Path(__file__).resolve().parent / "model_assets" / "GenD"
CLIP_DIR = MODEL_DIR / "clip-vit-large-patch14"

CLIP_REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VERIFRAME - GEND AND CLIP MODEL DOWNLOAD")
    print("=" * 60)

    print(f"\nGenD model: {GEND_MODEL_ID}")
    print(f"GenD revision: {GEND_MODEL_REVISION}")

    snapshot_download(
        repo_id=GEND_MODEL_ID,
        revision=GEND_MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        allow_patterns=["config.json", "model.safetensors"],
    )

    gend_config = MODEL_DIR / "config.json"
    gend_checkpoint = MODEL_DIR / "model.safetensors"

    if not gend_config.is_file():
        raise FileNotFoundError("GenD config.json download failed.")

    if not gend_checkpoint.is_file():
        raise FileNotFoundError("GenD model.safetensors download failed.")

    if gend_checkpoint.stat().st_size < 1_000_000_000:
        raise RuntimeError("GenD checkpoint appears incomplete.")

    print("\nGenD checkpoint verified.")
    print(f"GenD checkpoint size: {gend_checkpoint.stat().st_size:,} bytes")

    print(f"\nCLIP model: {CLIP_MODEL_ID}")
    print(f"CLIP revision: {CLIP_MODEL_REVISION}")
    print(f"CLIP destination: {CLIP_DIR}")

    snapshot_download(
        repo_id=CLIP_MODEL_ID,
        revision=CLIP_MODEL_REVISION,
        local_dir=str(CLIP_DIR),
        allow_patterns=CLIP_REQUIRED_FILES,
    )

    missing = [
        name for name in CLIP_REQUIRED_FILES
        if not (CLIP_DIR / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"CLIP download is incomplete. Missing files: {missing}"
        )

    clip_checkpoint = CLIP_DIR / "model.safetensors"

    if clip_checkpoint.stat().st_size < 1_000_000_000:
        raise RuntimeError("CLIP checkpoint appears incomplete.")

    print("\nCLIP backbone verified.")
    print(f"CLIP checkpoint size: {clip_checkpoint.stat().st_size:,} bytes")

    print("\n" + "=" * 60)
    print("ALL GEND MODEL ASSETS VERIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()
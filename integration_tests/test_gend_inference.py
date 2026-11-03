import os
import sys
import torch
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEND_ROOT = Path(os.getenv("VERIFRAME_GEND_ROOT", str(ROOT / "backend" / "model_sources" / "gend"))).expanduser().resolve()
IMAGE_PATH = Path(os.environ["VERIFRAME_GEND_TEST_IMAGE"]).expanduser().resolve()

MODEL_ID = "yermandy/GenD_CLIP_L_14"

if not GEND_ROOT.is_dir():
    raise FileNotFoundError(f"GenD source directory not found: {GEND_ROOT}")
if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")
sys.path.insert(0, str(GEND_ROOT))
os.chdir(GEND_ROOT)

from src.hf.modeling_gend import GenD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

print("Loading GenD directly from Hugging Face...")
model = GenD.from_pretrained(
    MODEL_ID,
    local_files_only=True
)

model.to(device)
model.eval()

print("Preparing image with official GenD preprocessing...")
image = Image.open(IMAGE_PATH).convert("RGB")

preprocess = model.feature_extractor.preprocess
input_tensor = preprocess(image).unsqueeze(0).to(device)

print("Running inference...")

with torch.inference_mode():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

real_score = probabilities[0].item()
fake_score = probabilities[1].item()

print("\n========== GEND INFERENCE RESULT ==========")
print(f"Device: {device}")
print(f"Real probability: {real_score:.6f}")
print(f"Fake probability: {fake_score:.6f}")
print(f"Prediction: {'FAKE' if fake_score >= real_score else 'REAL'}")
print("===========================================")
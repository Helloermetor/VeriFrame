import importlib
import torch

packages = [
    "torch", "torchvision", "transformers", "numpy",
    "cv2", "PIL", "sklearn", "skimage", "yaml",
    "timm", "lightning", "av", "imageio", "dlib",
    "albumentations", "einops", "tqdm"
]

print("Python package checks:")
for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "OK")
        print(f"{package}: OK ({version})")
    except Exception as e:
        print(f"{package}: FAILED ({type(e).__name__}: {e})")

print("\nCUDA:")
print("Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

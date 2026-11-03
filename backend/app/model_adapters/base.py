from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np

@dataclass
class FrameScore:
    logit: float
    raw_score: float
    embedding: np.ndarray | None = None

class DetectorAdapter(Protocol):
    name: str
    version: str
    input_size: tuple[int, int]

    def preprocess(self, face_rgb: np.ndarray):
        """Return a model-specific tensor using the checkpoint's exact preprocessing."""
        ...

    def predict(self, batch) -> Sequence[FrameScore]:
        ...

    def explain(self, face_rgb: np.ndarray, method: str = "gradcam++") -> np.ndarray:
        ...

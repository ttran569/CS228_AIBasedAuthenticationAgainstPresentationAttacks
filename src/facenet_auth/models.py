from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class FaceEmbedding:
    embedding: Sequence[float]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    if len(a) != len(b):
        raise ValueError("Embeddings must have the same length.")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean distance between two embeddings."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class FaceNetIdentityModel:
    """Wrapper around the facenet-pytorch FaceNet implementation."""

    def __init__(self, device: Optional[str] = None, *, image_size: int = 160, **detector_kwargs):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.detector_kwargs = {"image_size": image_size, "post_process": True, **detector_kwargs}

        try:
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ImportError as exc:  # pragma: no cover - exercised when package is absent
            raise ImportError(
                "facenet-pytorch is required for FaceNetIdentityModel. "
                "Install it with: pip install facenet-pytorch"
            ) from exc

        self.MTCNN = MTCNN
        self.InceptionResnetV1 = InceptionResnetV1
        self.detector = MTCNN(device=self.device.type, **self.detector_kwargs)
        self.model = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()

    def embed(self, face_image):
        """Detect, align, and embed a face image using the pretrained FaceNet model."""
        face = self.detector(face_image)
        if face is None:
            raise ValueError("No face was detected in the input image.")

        face_tensor = face.to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor.unsqueeze(0))
        return embedding.squeeze(0).cpu().numpy()


class _FallbackMiniFASNetV2(torch.nn.Module):
    """Fallback model used when the external MiniFASNet repository is unavailable."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3 * 224 * 224, 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        flat = x.reshape(x.size(0), -1)
        return self.head(flat)


class MiniFASNetV2PAD:
    """Wrapper around the MiniFASNetV2 anti-spoofing model used for PAD scoring."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model()
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def _load_model(self):
        candidate_modules = [
            "model_lib.MiniFASNet",
            "SilentFaceAntiSpoofing.model_lib.MiniFASNet",
            "src.model_lib.MiniFASNet",
            "MiniFASNet",
        ]

        last_error = None
        for module_name in candidate_modules:
            try:
                module = importlib.import_module(module_name)
                model_cls = getattr(module, "MiniFASNetV2", None)
                if model_cls is not None:
                    return model_cls()
            except Exception as exc:  # pragma: no cover - import path depends on external repo layout
                last_error = exc

        return _FallbackMiniFASNetV2()

    def score(self, face_image):
        """Return a bona-fide probability for a face crop using the PAD model."""
        if isinstance(face_image, torch.Tensor):
            tensor = face_image
        else:
            tensor = torch.tensor(face_image, dtype=torch.float32)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if hasattr(logits, "shape") and logits.ndim > 0:
                logits = logits.squeeze(-1) if logits.ndim > 1 else logits
            prob = torch.sigmoid(logits)

        if prob.numel() == 1:
            return float(prob.item())
        return float(prob.mean().item())

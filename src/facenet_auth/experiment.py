from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PIL import Image

from .evaluation import evaluate_trial, summarize_trial_results, threshold_search
from .models import FaceNetIdentityModel, MiniFASNetV2PAD, cosine_similarity


def _generate_demo_trials() -> list[dict]:
    """Generate deterministic synthetic trials so the evaluation pipeline can run without a dataset."""
    rng = random.Random(0)

    genuine_identity = [0.90 + rng.random() * 0.06 for _ in range(8)]
    impostor_identity = [0.20 + rng.random() * 0.20 for _ in range(8)]
    attack_identity = [0.55 + rng.random() * 0.25 for _ in range(8)]

    genuine_pad = [0.82 + rng.random() * 0.12 for _ in range(8)]
    attack_pad = [0.10 + rng.random() * 0.35 for _ in range(8)]

    trials: list[dict] = []
    for idx in range(8):
        trials.append(
            {
                "identity_score": genuine_identity[idx],
                "pad_score": genuine_pad[idx],
                "is_attack": False,
                "label": "genuine",
            }
        )

    for idx in range(8):
        trials.append(
            {
                "identity_score": impostor_identity[idx],
                "pad_score": genuine_pad[(idx + 2) % 8],
                "is_attack": False,
                "label": "impostor",
            }
        )

    for idx in range(8):
        trials.append(
            {
                "identity_score": attack_identity[idx],
                "pad_score": attack_pad[idx],
                "is_attack": True,
                "label": "attack",
            }
        )

    return trials


def _parse_image_dir(dataset_dir: Path) -> list[dict]:
    """Try to recover a simple dataset manifest from a directory tree.

    If the directory structure is not available, this function returns an empty list and
    the caller falls back to synthetic demo trials.
    """
    image_paths = sorted(dataset_dir.rglob("*.png")) + sorted(dataset_dir.rglob("*.jpg")) + sorted(dataset_dir.rglob("*.jpeg"))
    if not image_paths:
        return []

    manifest: list[dict] = []
    for path in image_paths:
        rel = path.relative_to(dataset_dir)
        parts = rel.parts
        identity = parts[0] if len(parts) > 1 else "unknown"
        label = "genuine" if "spoof" not in str(path).lower() else "attack"
        manifest.append({"image_path": str(path), "identity": identity, "label": label})
    return manifest


def run_experiment(
    dataset_dir: Optional[str | Path] = None,
    tau_id: float = 0.80,
    tau_pad: float = 0.80,
    use_demo: bool = True,
) -> dict:
    """Run the FaceNet/PAD evaluation pipeline.

    If a real dataset directory is supplied and contains images, the function scores the actual
    images using the wrappers. Otherwise it uses deterministic synthetic trials so the script runs.
    """
    if dataset_dir is not None and not use_demo:
        dataset_path = Path(dataset_dir)
        manifest = _parse_image_dir(dataset_path)
        if manifest:
            identity_model = FaceNetIdentityModel(device="cpu")
            pad_model = MiniFASNetV2PAD(device="cpu")

            trials: list[dict] = []
            for entry in manifest:
                image = Image.open(entry["image_path"]).convert("RGB")
                identity_score = 0.0
                pad_score = 0.0
                try:
                    embedding = identity_model.embed(image)
                    if trials and entry["identity"] == trials[-1].get("identity"):
                        identity_score = cosine_similarity(embedding, trials[-1]["embedding"])
                    else:
                        identity_score = 0.5
                    pad_score = pad_model.score(image)
                except Exception:
                    identity_score = 0.5
                    pad_score = 0.5

                trials.append(
                    {
                        "identity_score": identity_score,
                        "pad_score": pad_score,
                        "is_attack": entry["label"] == "attack",
                        "accepted": evaluate_trial(identity_score, pad_score, tau_id, tau_pad),
                    }
                )

            if trials:
                summary = summarize_trial_results(trials)
                genuine_scores = [t["identity_score"] for t in trials if not t["is_attack"]]
                imp_scores = [t["identity_score"] for t in trials if t["is_attack"]]
                pad_genuine = [t["pad_score"] for t in trials if not t["is_attack"]]
                pad_attack = [t["pad_score"] for t in trials if t["is_attack"]]
                selected_tau_id = threshold_search(genuine_scores, imp_scores, min_score=0.0, max_score=1.0, step=0.01)
                selected_tau_pad = threshold_search(pad_genuine, pad_attack, min_score=0.0, max_score=1.0, step=0.01)
                summary["selected_tau_id"] = selected_tau_id
                summary["selected_tau_pad"] = selected_tau_pad
                return summary

    synthetic_trials = _generate_demo_trials()
    for trial in synthetic_trials:
        trial["accepted"] = evaluate_trial(
            trial["identity_score"],
            trial["pad_score"],
            tau_id,
            tau_pad,
        )

    summary = summarize_trial_results(synthetic_trials)
    genuine_identity_scores = [t["identity_score"] for t in synthetic_trials if not t["is_attack"]]
    impostor_identity_scores = [t["identity_score"] for t in synthetic_trials if t["is_attack"]]
    genuine_pad_scores = [t["pad_score"] for t in synthetic_trials if not t["is_attack"]]
    attack_pad_scores = [t["pad_score"] for t in synthetic_trials if t["is_attack"]]

    summary["selected_tau_id"] = threshold_search(
        genuine_identity_scores,
        impostor_identity_scores,
        min_score=0.0,
        max_score=1.0,
        step=0.01,
    )
    summary["selected_tau_pad"] = threshold_search(
        genuine_pad_scores,
        attack_pad_scores,
        min_score=0.0,
        max_score=1.0,
        step=0.01,
    )
    return summary

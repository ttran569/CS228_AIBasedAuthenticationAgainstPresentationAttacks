from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facenet_auth.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Notebook-centered FaceNet + MiniFASNetV2 evaluation for CelebA-Spoof. "
            "The canonical implementation is in CS228_facenet_pytorch_baseline_MiniFASNetV2_PAD-2.ipynb."
        )
    )
    parser.add_argument("--tau-id", type=float, default=0.80, help="Identity verification threshold")
    parser.add_argument("--tau-pad", type=float, default=0.80, help="PAD acceptance threshold")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional dataset directory containing images")
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo evaluation even if a dataset dir is present")
    args = parser.parse_args()

    print("Project workflow: FaceNet baseline + MiniFASNetV2 PAD on CelebA-Spoof")
    print("Canonical notebook: CS228_facenet_pytorch_baseline_MiniFASNetV2_PAD-2.ipynb")

    summary = run_experiment(
        dataset_dir=args.dataset_dir,
        tau_id=args.tau_id,
        tau_pad=args.tau_pad,
        use_demo=args.demo,
    )

    print(f"Selected identity threshold: {summary['selected_tau_id']:.2f}")
    print(f"Selected PAD threshold: {summary['selected_tau_pad']:.2f}")
    print(summary)


if __name__ == "__main__":
    main()

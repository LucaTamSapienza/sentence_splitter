"""
Download the fine-tuned checkpoint and (optionally) the training data from HuggingFace.

Usage:
    python download_model.py              # download model only
    python download_model.py --data       # download model + dataset
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).parent
MODEL_REPO = "Famezz/xlmr-sentence-splitter"
DATA_REPO = "Famezz/sentence-splitter-ud-data"


def download_model():
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / "best_xlmr_model.pt"

    if dest.exists():
        print(f"Checkpoint already exists at {dest}")
        return

    print(f"Downloading model from {MODEL_REPO}...")
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_xlmr_model.pt",
        local_dir=ckpt_dir,
    )
    print(f"Saved to {dest}")


def download_data():
    # Check if UD dirs already exist
    existing = list(ROOT.glob("UD_*"))
    if existing:
        print(f"Dataset already present ({len(existing)} UD directories)")
        return

    print(f"Downloading dataset from {DATA_REPO}...")
    snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        local_dir=ROOT,
    )
    print("Dataset downloaded.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", help="Also download the UD training/test data")
    args = parser.parse_args()

    download_model()
    if args.data:
        download_data()


if __name__ == "__main__":
    main()

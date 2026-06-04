"""
Download the local NER model required by Local PII Guard.

Downloads Qwen2.5-3B-Instruct-IQ3_M.gguf (~1.5 GB) from HuggingFace.
The model is used by LocalNameExtractor in pii_llm_names.py for name detection.

Usage:
    python download_model.py
    python download_model.py --output /path/to/models/
    python download_model.py --check
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF"
    "/resolve/main/Qwen2.5-3B-Instruct-IQ3_M.gguf?download=true"
)
MODEL_FILENAME = "Qwen2.5-3B-Instruct-IQ3_M.gguf"
MODEL_SIZE_APPROX_MB = 1500


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, int(100 * downloaded / total_size))
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        bar = "#" * (percent // 5) + "-" * (20 - percent // 5)
        sys.stdout.write(f"\r[{bar}] {percent:3d}%  {downloaded_mb:.1f} / {total_mb:.1f} MB")
    else:
        sys.stdout.write(f"\r{downloaded / (1024 * 1024):.1f} MB downloaded")
    sys.stdout.flush()


def download_model(output_dir: str = ".") -> Path:
    """Download the model to output_dir. Returns the path to the downloaded file."""
    dest = Path(output_dir) / MODEL_FILENAME

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {dest} ({size_mb:.0f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_FILENAME} (~{MODEL_SIZE_APPROX_MB} MB) to {dest}")
    print("This may take several minutes on a slow connection.\n")

    try:
        urllib.request.urlretrieve(MODEL_URL, str(dest), _progress_hook)
        print(f"\nDownload complete: {dest}")
        return dest
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def check_model(output_dir: str = ".") -> bool:
    """Return True if the model file exists and has a plausible size."""
    dest = Path(output_dir) / MODEL_FILENAME
    if not dest.exists():
        print(f"Model not found: {dest}")
        return False
    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb < 100:
        print(f"Model file looks incomplete: {dest} ({size_mb:.0f} MB)")
        return False
    print(f"Model OK: {dest} ({size_mb:.0f} MB)")
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the Qwen2.5-3B GGUF model for Local PII Guard name detection."
    )
    p.add_argument(
        "--output", "-o",
        default=".",
        help="Directory to save the model file (default: current directory)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Check whether the model already exists without downloading",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.check:
        ok = check_model(args.output)
        sys.exit(0 if ok else 1)
    else:
        download_model(args.output)

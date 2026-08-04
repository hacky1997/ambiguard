"""Fetch or validate CenterDistill checkpoint.

Checkpoint resolution order (spec §5.1):
    1. Local dir at settings.gate_checkpoint_path
    2. HF Hub via settings.gate_hf_repo → snapshot_download
    3. Absent → prints instructions, exits cleanly

Expected checkpoint contents:
    config.json          — model config
    model.safetensors    — weights
    tokenizer files      — tokenizer config, vocab, etc.
    centers.npy          — K×768 centroids from spectral clustering

Usage:
    python scripts/fetch_checkpoint.py [--validate-only]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.settings import get_settings

logger = logging.getLogger(__name__)

_REQUIRED_FILES: list[str] = [
    "config.json",
    "centers.npy",
]

_EXPECTED_CENTER_SHAPE: tuple[int, int] = (5, 768)  # K=5, LaBSE dim=768


def _validate_checkpoint(checkpoint_dir: Path) -> bool:
    """Validate that a checkpoint directory has the required files."""
    full_pt = checkpoint_dir / "centerdistill_full.pt"
    if full_pt.exists():
        logger.info("Found repaired artifact: %s ✓", full_pt)
        return True

    all_ok = True
    for fname in _REQUIRED_FILES:
        fpath = checkpoint_dir / fname
        if not fpath.exists():
            logger.error("Missing required file: %s", fpath)
            all_ok = False

    # Validate centers shape
    centers_path = checkpoint_dir / "centers.npy"
    if centers_path.exists():
        try:
            centers: np.ndarray = np.load(str(centers_path))
            if centers.shape != _EXPECTED_CENTER_SHAPE:
                logger.error(
                    "centers.npy has shape %s, expected %s",
                    centers.shape,
                    _EXPECTED_CENTER_SHAPE,
                )
                all_ok = False
            else:
                logger.info("centers.npy shape OK: %s", centers.shape)
        except Exception as exc:
            logger.error("Failed to load centers.npy: %s", exc)
            all_ok = False

    # Check for model weights
    has_weights = any(
        (checkpoint_dir / f).exists()
        for f in ["model.safetensors", "pytorch_model.bin"]
    )
    if not has_weights:
        logger.warning(
            "No model weights found (model.safetensors or pytorch_model.bin). "
            "The gate will use heuristic fallback."
        )

    return all_ok


def _download_from_hub(repo_id: str, target_dir: Path | None = None) -> Path:
    """Download checkpoint from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Install with: pip install huggingface-hub"
        )
        sys.exit(1)

    logger.info("Downloading checkpoint from HF Hub: %s", repo_id)
    local_dir: str = snapshot_download(
        repo_id,
        local_dir=str(target_dir) if target_dir else None,
    )
    logger.info("Downloaded to: %s", local_dir)
    return Path(local_dir)


def main() -> None:
    """Fetch or validate the CenterDistill checkpoint."""
    parser = argparse.ArgumentParser(
        description="Fetch or validate CenterDistill checkpoint"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing checkpoint, don't download",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Check local path first
    if settings.gate_checkpoint_path:
        cp_path = settings.gate_checkpoint_path
        if cp_path.exists():
            logger.info("Local checkpoint found: %s", cp_path)
            if _validate_checkpoint(cp_path):
                logger.info("Checkpoint valid ✓")
            else:
                logger.error("Checkpoint validation failed ✗")
                sys.exit(1)
            return
        elif args.validate_only:
            logger.error("Checkpoint path configured but not found: %s", cp_path)
            sys.exit(1)

    # Try HF Hub
    if settings.gate_hf_repo and not args.validate_only:
        target = settings.gate_checkpoint_path or Path("checkpoints/centerdistill")
        cp_path = _download_from_hub(settings.gate_hf_repo, target)
        if _validate_checkpoint(cp_path):
            logger.info("Checkpoint valid ✓")
        else:
            logger.error("Downloaded checkpoint validation failed ✗")
            sys.exit(1)
        return

    # No checkpoint available
    print(
        "\n"
        "No CenterDistill checkpoint configured.\n"
        "\n"
        "The app will use the heuristic fallback gate (fallback_used=True).\n"
        "This is fine for development and demo — the comparison harness will\n"
        "label fallback results accordingly.\n"
        "\n"
        "To use the learned gate, set one of:\n"
        "  GATE_CHECKPOINT_PATH=/path/to/checkpoint\n"
        "  GATE_HF_REPO=your-username/centerdistill-checkpoint\n"
        "\n"
        "Expected checkpoint files:\n"
        "  config.json, model.safetensors, tokenizer files, centers.npy\n"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()

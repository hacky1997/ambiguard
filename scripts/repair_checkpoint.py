#!/usr/bin/env python3
"""
Repair the CenterDistill checkpoint so the gate can actually run.

WHY THIS EXISTS
---------------
The training notebook saved the model correctly, then a later cell rebuilt it as a
plain HuggingFace QA model and called save_pretrained() over the same directory.
That rebuild copied the encoder and the span head but NOT the center head, so the
top-level model.safetensors is span-only. The 5-way center head — the entire gate —
is absent from it.

The intermediate Trainer checkpoints (checkpoint-*/) were written before that
overwrite and still contain the full state dict, center head included.

This script extracts from the newest Trainer checkpoint and writes a single clean
artifact the gate adapter can load, with no notebook context and no clustering
artifacts required.

USAGE
-----
    python scripts/repair_checkpoint.py \
        --checkpoint-dir  /path/to/centerdistill \
        --base-model      /path/to/baseline_en \
        --out             ./checkpoints/centerdistill_full

    # --base-model may also be a hub id, e.g. deepset/xlm-roberta-large-squad2

The output directory contains:
    centerdistill_full.pt   state dict + K + thresholds + base model reference
    tokenizer files
    manifest.json           provenance: which checkpoint, which keys, hashes
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Thresholds from the paper. tau_ent is in NATS.
DEFAULT_THRESHOLDS = {"tau_conf": 0.44, "tau_ent": 1.51, "tau_multi": 0.24}


class CenterDistillModel(nn.Module):
    """Mirrors the training-time class. Head names must match the saved state dict."""

    def __init__(self, base_name: str, num_centers: int) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(base_name)
        hidden = self.encoder.config.hidden_size
        self.span_head = nn.Linear(hidden, 2)
        self.center_head = nn.Linear(hidden, num_centers)
        self.num_centers = num_centers


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    st_file = path / "model.safetensors"
    if st_file.exists():
        from safetensors.torch import load_file

        return load_file(str(st_file), device="cpu")
    bin_file = path / "pytorch_model.bin"
    if bin_file.exists():
        return torch.load(bin_file, map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing checkpoint-*/ subdirectories",
    )
    ap.add_argument(
        "--base-model",
        required=True,
        help="Path or hub id of the encoder the model was built from",
    )
    ap.add_argument("--out", default="./checkpoints/centerdistill_full")
    ap.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific checkpoint step. Default: highest.",
    )
    ap.add_argument(
        "--thresholds",
        default=None,
        help="JSON file with tau_conf / tau_ent / tau_multi. Default: paper values.",
    )
    args = ap.parse_args()

    ckpt_root = Path(args.checkpoint_dir)
    out_dir = Path(args.out)

    # ── 1. Locate the checkpoint ───────────────────────────────────────────────
    candidates = sorted(
        (Path(p) for p in glob.glob(str(ckpt_root / "checkpoint-*"))),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    if not candidates:
        print(
            f"ERROR: no checkpoint-* directories under {ckpt_root}", file=sys.stderr
        )
        print(
            "The top-level model.safetensors is span-only and cannot be used.",
            file=sys.stderr,
        )
        return 1

    if args.step is not None:
        matches = [p for p in candidates if p.name == f"checkpoint-{args.step}"]
        if not matches:
            print(
                f"ERROR: checkpoint-{args.step} not found. "
                f"Available: {[p.name for p in candidates]}",
                file=sys.stderr,
            )
            return 1
        ckpt = matches[0]
    else:
        ckpt = candidates[-1]

    print(f"Checkpoints found : {[p.name for p in candidates]}")
    print(f"Using             : {ckpt.name}")

    # ── 2. Load and inspect ────────────────────────────────────────────────────
    state = _load_state(ckpt)
    state.pop("mean_soft_labels", None)  # trainer buffer, not a parameter

    center_keys = sorted(k for k in state if "center_head" in k)
    if not center_keys:
        print(f"\nERROR: no center_head.* keys in {ckpt}", file=sys.stderr)
        print(f"Keys present (sample): {sorted(state)[:10]}", file=sys.stderr)
        print(
            "\nThis checkpoint has no gate. Try an earlier step, or the center "
            "head was never persisted and retraining is required.",
            file=sys.stderr,
        )
        return 1

    w = state["center_head.weight"]
    K = int(w.shape[0])
    print(f"\ncenter_head keys  : {center_keys}")
    print(f"center_head shape : {tuple(w.shape)}  ->  K = {K}")

    if K < 2:
        print(f"ERROR: K={K} is not a usable number of centers", file=sys.stderr)
        return 1

    # Degenerate-head check: an untrained head has near-zero weights.
    wnorm = float(w.norm())
    print(f"center_head |W|   : {wnorm:.4f}")
    if wnorm < 1e-3:
        print(
            "WARNING: center head weights are near zero — it may never have "
            "been trained. Verify before trusting any gate output.",
            file=sys.stderr,
        )

    # ── 3. Rebuild and load ────────────────────────────────────────────────────
    print(f"\nRebuilding model from base: {args.base_model}")
    model = CenterDistillModel(args.base_model, K)
    missing, unexpected = model.load_state_dict(state, strict=False)

    missing_center = [m for m in missing if "center_head" in m]
    if missing_center:
        print(f"ERROR: center head failed to load: {missing_center}", file=sys.stderr)
        return 1

    print(
        f"  missing    : {len(missing)} keys"
        f"{' (' + ', '.join(missing[:4]) + ')' if missing else ''}"
    )
    print(
        f"  unexpected : {len(unexpected)} keys"
        f"{' (' + ', '.join(unexpected[:4]) + ')' if unexpected else ''}"
    )
    model.eval()

    # ── 4. Smoke test — a real forward pass through the head ───────────────────
    from transformers import AutoTokenizer

    tok_src = ckpt if (ckpt / "tokenizer_config.json").exists() else ckpt_root
    tok = AutoTokenizer.from_pretrained(str(tok_src))

    with torch.inference_mode():
        enc = tok(
            "what are the side effects",
            "Medication A causes nausea. Medication B causes headaches.",
            truncation="only_second",
            max_length=384,
            padding="max_length",
            return_tensors="pt",
        )
        cls = model.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            return_dict=True,
        ).last_hidden_state[:, 0, :]
        probs = torch.softmax(model.center_head(cls), dim=-1)[0]

    p = probs.tolist()
    entropy = float(-(probs * torch.log(probs + 1e-10)).sum())  # NATS
    print(f"\nSmoke test")
    print(f"  P_S      : {[round(x, 4) for x in p]}")
    print(f"  sum      : {sum(p):.6f}")
    print(f"  max      : {max(p):.4f}")
    print(
        f"  entropy  : {entropy:.4f} nats  (max possible {torch.log(torch.tensor(float(K))):.4f})"
    )

    if abs(sum(p) - 1.0) > 1e-4:
        print("ERROR: distribution does not sum to 1", file=sys.stderr)
        return 1

    uniform = 1.0 / K
    if max(abs(x - uniform) for x in p) < 0.01:
        print(
            "WARNING: output is essentially uniform. The head may be untrained "
            "or the encoder weights did not load correctly.",
            file=sys.stderr,
        )

    # ── 5. Thresholds ──────────────────────────────────────────────────────────
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)
        thresholds = {k: float(thresholds[k]) for k in DEFAULT_THRESHOLDS}
        print(f"\nThresholds (from {args.thresholds}): {thresholds}")
    else:
        thresholds = dict(DEFAULT_THRESHOLDS)
        print(f"\nThresholds (paper defaults): {thresholds}")
    print("  NOTE: tau_ent is in NATS. Entropy must be computed with natural log.")

    # ── 6. Write output ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "centerdistill_full.pt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "K": K,
            "base_model": args.base_model,
            "thresholds": thresholds,
            "format_version": 1,
        },
        artifact,
    )
    tok.save_pretrained(str(out_dir))

    manifest = {
        "source_checkpoint": str(ckpt),
        "source_step": int(ckpt.name.rsplit("-", 1)[1]),
        "available_steps": [int(p.name.rsplit("-", 1)[1]) for p in candidates],
        "base_model": args.base_model,
        "K": K,
        "center_head_keys": center_keys,
        "center_head_weight_norm": round(wnorm, 6),
        "thresholds": thresholds,
        "entropy_units": "nats",
        "artifact_sha256": _sha256(artifact),
        "smoke_test_distribution": [round(x, 6) for x in p],
        "note": (
            "Extracted from a Trainer checkpoint because the top-level "
            "model.safetensors was overwritten with a span-only HF QA model and "
            "does not contain the center head."
        ),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {artifact}")
    print(f"Wrote {out_dir / 'manifest.json'}")
    print(f"\nPoint the gate at it:\n  AMBIGUARD_GATE_CHECKPOINT_PATH={out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

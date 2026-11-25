#!/usr/bin/env python3
"""Verify that the exported model is exactly the retained final checkpoint.

Run this on the Rockfish MACE environment.  It does not mutate either input.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--compiled-model", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--expected-epoch", type=int, default=999)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    candidates = sorted(args.checkpoint_dir.glob(f"*_epoch-{args.expected_epoch}.pt"))
    result = {
        "model_path": str(args.model),
        "model_sha256": sha256(args.model),
        "compiled_model_path": str(args.compiled_model),
        "compiled_model_sha256": sha256(args.compiled_model),
        "checkpoint_candidates": [str(p) for p in candidates],
        "expected_epoch": args.expected_epoch,
        "status": "started",
    }
    if len(candidates) != 1:
        result.update({"status": "failed", "reason": f"expected exactly one epoch-{args.expected_epoch} checkpoint"})
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(2)

    checkpoint_path = candidates[0]
    result["checkpoint_path"] = str(checkpoint_path)
    result["checkpoint_sha256"] = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exported = torch.load(args.model, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["model"]
    exported_state = exported.state_dict()
    keys_match = sorted(checkpoint_state) == sorted(exported_state)
    max_abs = 0.0
    mismatched = []
    if keys_match:
        for key in sorted(checkpoint_state):
            a = checkpoint_state[key].detach().cpu()
            b = exported_state[key].detach().cpu()
            delta = float((a - b).abs().max().item()) if a.numel() else 0.0
            max_abs = max(max_abs, delta)
            if delta != 0.0:
                mismatched.append({"key": key, "max_abs": delta})
    result.update(
        {
            "checkpoint_model_tensor_digest": tensor_digest(checkpoint_state),
            "exported_model_tensor_digest": tensor_digest(exported_state),
            "state_dict_keys_match": keys_match,
            "state_dict_max_abs_difference": max_abs,
            "state_dict_exact_match": bool(keys_match and max_abs == 0.0),
            "checkpoint_keys": len(checkpoint_state),
            "exported_keys": len(exported_state),
            "mismatched_tensor_count": len(mismatched),
            "mismatched_tensor_preview": mismatched[:10],
            "compiled_exists": args.compiled_model.exists(),
            "status": "complete" if keys_match and max_abs == 0.0 and args.compiled_model.exists() else "failed",
        }
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise SystemExit(3)


if __name__ == "__main__":
    main()

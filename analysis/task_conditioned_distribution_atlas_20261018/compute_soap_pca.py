#!/usr/bin/env python3
"""Compute the retained SOAP/PCA space used by the distribution atlas.

This script is intentionally small and provenance-first.  The PCA fit is made
on the grouped training split only.  Grouped reference files and MLIP
generated MD segments are transformed with the same fit; no model ranking is
performed here.  The script is suitable for the Rockfish MACE environment,
where ASE, DScribe, and scikit-learn are installed.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import iread
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA


SPECIES = ["Cr", "Sb", "Te"]
SOAP_KW = {
    "species": SPECIES,
    "periodic": True,
    "r_cut": 5.0,
    "n_max": 4,
    "l_max": 4,
    "sigma": 0.5,
    "average": "inner",
    "sparse": False,
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["grouped", "md"], required=True)
    p.add_argument("--train", type=Path)
    p.add_argument("--valid", type=Path)
    p.add_argument("--test", type=Path)
    p.add_argument("--neb", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--pca", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--family", default="")
    p.add_argument("--model-id", default="")
    p.add_argument("--training-seed", default="")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--guard-short-contact", type=float, default=0.0)
    return p


def make_soap() -> SOAP:
    return SOAP(**SOAP_KW)


def descriptor_rows(path: Path, soap: SOAP, *, start: int = 0, stop: int = 0,
                    stride: int = 1, max_frames: int = 0,
                    guard_short_contact: float = 0.0):
    if stop <= 0 and max_frames:
        stop = start + max_frames * stride
    for i, atoms in enumerate(iread(str(path), index=":")):
        if i < start:
            continue
        if stop > 0 and i >= stop:
            break
        if (i - start) % stride:
            continue
        frame = i
        min_distance = np.nan
        if guard_short_contact > 0:
            _, _, distances = neighbor_list("ijd", atoms, guard_short_contact)
            if len(distances):
                min_distance = float(np.min(distances))
                yield frame, atoms, None, min_distance
                continue
        desc = np.asarray(soap.create(atoms), dtype=np.float64).reshape(-1)
        yield frame, atoms, desc, min_distance
        del desc
        gc.collect()


def fit_grouped(args: argparse.Namespace) -> None:
    for required in [args.train, args.valid, args.test, args.neb]:
        if required is None:
            raise ValueError("grouped mode requires --train, --valid, --test, and --neb")
    soap = make_soap()
    descs: list[np.ndarray] = []
    rows: list[dict] = []
    split_paths = [("train", args.train), ("valid", args.valid), ("test", args.test), ("neb", args.neb)]
    for split, path in split_paths:
        for frame, atoms, desc, _ in descriptor_rows(path, soap):
            descs.append(desc)
            info = dict(atoms.info)
            rows.append({
                "split": split,
                "frame": frame,
                "natoms": len(atoms),
                "source_trajectory": info.get("source_trajectory", ""),
                "source_file": info.get("source_file", str(path)),
            })
    X = np.vstack(descs)
    train_mask = np.array([r["split"] == "train" for r in rows])
    mean = X[train_mask].mean(axis=0)
    scale = X[train_mask].std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    pca = PCA(n_components=2, whiten=True, random_state=0)
    coords = pca.fit_transform(Z)
    for row, coord in zip(rows, coords):
        row["pca1"] = float(coord[0])
        row["pca2"] = float(coord[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    pca_path = args.output.with_name("soap_pca_fit.npz")
    np.savez_compressed(
        pca_path,
        mean=mean,
        scale=scale,
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        components=np.asarray(pca.components_, dtype=np.float64),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )
    meta = {
        "soap": SOAP_KW,
        "fit_split": "train",
        "n_rows": len(rows),
        "n_train": int(train_mask.sum()),
        "n_valid": int((np.array([r["split"] for r in rows]) == "valid").sum()),
        "n_test": int((np.array([r["split"] for r in rows]) == "test").sum()),
        "n_neb": int((np.array([r["split"] for r in rows]) == "neb").sum()),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    args.output.with_name("soap_pca_fit.json").write_text(json.dumps(meta, indent=2) + "\n")


def transform_md(args: argparse.Namespace) -> None:
    if args.input is None or args.pca is None:
        raise ValueError("md mode requires --input and --pca")
    fit = np.load(args.pca)
    soap = make_soap()
    rows: list[dict] = []
    for frame, atoms, desc, min_distance in descriptor_rows(args.input, soap, start=args.start, stop=args.stop,
                                                             stride=args.stride, max_frames=args.max_frames,
                                                             guard_short_contact=args.guard_short_contact):
        if desc is None:
            coord = (np.nan, np.nan)
            status = "short_contact_skipped"
        else:
            z = (desc - fit["mean"]) / fit["scale"]
            coord = fit["components"] @ (z - fit["pca_mean"])
            status = "ok"
        rows.append({
            "family": args.family,
            "model_id": args.model_id,
            "training_seed": args.training_seed,
            "source_file": str(args.input),
            "frame": frame,
            "step": atoms.info.get("step", frame),
            "time_ps": atoms.info.get("time_ps", np.nan),
            "pca1": float(coord[0]),
            "pca2": float(coord[1]),
            "natoms": len(atoms),
            "min_distance_A": min_distance,
            "soap_status": status,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps({
        "input": str(args.input), "output": str(args.output), "family": args.family,
        "model_id": args.model_id, "training_seed": args.training_seed,
        "start": args.start, "stop": args.stop, "stride": args.stride,
        "n_rows": len(rows), "n_soap_ok": int(sum(r["soap_status"] == "ok" for r in rows)),
        "n_short_contact_skipped": int(sum(r["soap_status"] != "ok" for r in rows)), "soap": SOAP_KW,
    }, indent=2) + "\n")


def main() -> None:
    args = parser().parse_args()
    if args.mode == "grouped":
        fit_grouped(args)
    else:
        transform_md(args)


if __name__ == "__main__":
    main()

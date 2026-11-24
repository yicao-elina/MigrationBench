from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--local-root", required=True)
ap.add_argument("--remote-root", required=True)
a = ap.parse_args()
d = pd.read_csv(a.input)
d["input_path"] = d["input_path"].str.replace(str(Path(a.local_root) / "inputs"), str(Path(a.remote_root) / "inputs"), regex=False)
d.to_csv(a.output, index=False)

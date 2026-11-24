"""Shared helpers for per-item reproduction scripts and their evidence files.

Every figure/table generator under items/<section>/<item>/source/ uses this module to
  * read inputs by repo-relative path (never absolute, never values typed into code),
  * write regenerated outputs under <item>/rerun_output/ (git-ignored),
  * compare them with what was submitted, and
  * write <item>/evidence/<name>.json: command, input/output sha256, check results.

Usage inside a generator:
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"))
    from repro_lib import Evidence, submitted_table, render_table
    ev = Evidence(__file__, "tab:resp-c10-grouped-split")
    data = json.loads(ev.input("paper/revision1/data_processed/current_release/x.json").read_text())
    rows = [...]                                           # computed from data only
    ev.compare_table(rows, "response_to_reviewers.tex")     # numbers at submitted precision
    raise SystemExit(ev.write())
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
PAPER = REPO / "paper/revision1"
NUMBER = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)(?:\s*\\times\s*10\^\{?(-?\d+)\}?)?")
ROUNDING_SLACK = 1e-9
SKIP_LINE = re.compile(r"^\\(toprule|midrule|bottomrule|cmidrule|hline|multicolumn|addlinespace)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO).as_posix()


# ---------------------------------------------------------------- LaTeX tables
def _input_path(name: str) -> Path | None:
    """Resolve an \\input path the way TEXINPUTS=.:..// does (paper/revision1, then paper/)."""
    name = name if name.endswith(".tex") else name + ".tex"
    return next((base / name for base in (PAPER, PAPER.parent) if (base / name).exists()), None)


def table_block(tex_name: str, label: str) -> str:
    """The tabular body of the table carrying \\label{label} in paper/revision1/<tex_name>.

    Follows a single \\input{...} inside the table environment (SI tables kept in tables/)."""
    text = (PAPER / tex_name).read_text()
    at = text.find("\\label{" + label + "}")
    if at < 0:  # the label may sit inside a file the document \input's
        for name in re.findall(r"^[^%\n]*\\input\{([^}]+)\}", text, re.M):
            candidate = _input_path(name)
            if candidate and "\\label{" + label + "}" in candidate.read_text():
                text = candidate.read_text()
                at = text.find("\\label{" + label + "}")
                break
    if at < 0:
        raise KeyError(f"{label} not found in {tex_name}")
    start, end = text.rfind("\\begin{table", 0, at), text.find("\\end{table", at)
    block = text[start:end]
    included = re.search(r"\\input\{([^}]+)\}", block)
    if "\\begin{tabular" not in block and included:
        block = _input_path(included.group(1)).read_text()
    match = re.search(r"\\begin\{tabular\*?\}(?:\{[^}]*\})+(.*?)\\end\{tabular", block, re.S)
    if not match:
        raise ValueError(f"no tabular for {label}")
    return match.group(1)


def table_rows(body: str) -> list[list[str]]:
    """Data rows (cells stripped) of a tabular body; rules and \\multicolumn lines dropped."""
    rows = []
    for line in body.replace("\n", " ").split("\\\\"):
        line = line.strip()
        while SKIP_LINE.match(line):
            line = re.sub(r"^\\(toprule|midrule|bottomrule|hline|addlinespace)(\[[^]]*\])?", "", line).strip()
            line = re.sub(r"^\\cmidrule(\([^)]*\))?\{[^}]*\}", "", line).strip()
            if line.startswith("\\multicolumn"):
                line = ""
        if line:
            rows.append([cell.strip() for cell in line.split("&")])
    return rows


def submitted_table(tex_name: str, label: str, header_rows: int = 1) -> list[list[str]]:
    rows = table_rows(table_block(tex_name, label))
    return rows[header_rows:]


def numbers(cell: str) -> list[tuple[float, float]]:
    """(value, half unit of the last displayed digit) for each number in a cell."""
    out = []
    cleaned = re.sub(r"\\texttt\{[^}]*\}|\$\^\{?[a-z]\}?\$|perplexity|Lmax\d", " ", cell)
    for mantissa, exponent in NUMBER.findall(cleaned):
        plain = mantissa.replace(",", "")
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        scale = 10.0 ** int(exponent) if exponent else 1.0
        out.append((float(plain) * scale, 0.5 * 10.0 ** -decimals * scale))
    return out


def norm_text(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("~", " ")).strip()


def compare_rows(submitted: list[list[str]], generated: list[list[str]]) -> list[str]:
    """Every number must equal the generated one at the submitted precision; text must match."""
    errors = []
    if len(submitted) != len(generated):
        return [f"{len(submitted)} submitted rows vs {len(generated)} generated"]
    for index, (sub, gen) in enumerate(zip(submitted, generated), start=1):
        if len(sub) != len(gen):
            errors.append(f"row {index}: {len(sub)} vs {len(gen)} cells: {sub} | {gen}")
            continue
        for col, (s_cell, g_cell) in enumerate(zip(sub, gen), start=1):
            s_num, g_num = numbers(s_cell), [n for n, _ in numbers(g_cell)]
            if not s_num and not g_num:
                if norm_text(s_cell) != norm_text(g_cell):
                    errors.append(f"row {index} col {col}: text {s_cell!r} vs {g_cell!r}")
                continue
            if len(s_num) != len(g_num):
                errors.append(f"row {index} col {col}: {s_cell!r} vs {g_cell!r} (number count)")
                continue
            for (value, half), produced in zip(s_num, g_num):
                if abs(value - produced) > half + ROUNDING_SLACK:
                    errors.append(f"row {index} col {col}: submitted {value} vs generated {produced}")
    return errors


def fmt_int(value) -> str:
    """1786 -> '1,786' (thousands separator as in the submitted tables)."""
    return f"{int(round(float(value))):,}"


def fmt_num(value, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def fmt_pm(mean, sd, decimals: int = 2, sd_decimals: int | None = None) -> str:
    """'$0.47\\pm0.12$'."""
    sd_decimals = decimals if sd_decimals is None else sd_decimals
    return f"${float(mean):.{decimals}f}\\pm{float(sd):.{sd_decimals}f}$"


def fmt_sci(value, digits: int = 2) -> str:
    """0.063 -> '$6.3\\times10^{-2}$' with `digits` significant figures."""
    mantissa, exponent = f"{float(value):.{digits - 1}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def render_table(rows: list[list[str]]) -> str:
    return "\n".join(" & ".join(cells) + r" \\" for cells in rows) + "\n"


# ------------------------------------------------------------------- figures
def rasterize(pdf: Path, dpi: int, page: int = 1) -> Path:
    out = Path(tempfile.mkdtemp()) / "page"
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", "-singlefile",
                    str(pdf), str(out)], check=True, capture_output=True)
    return out.with_suffix(".png")


def image_difference(a: Path, b: Path) -> dict:
    from PIL import Image, ImageChops

    first, second = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    if first.size != second.size:
        return {"same_size": False, "size_a": first.size, "size_b": second.size}
    diff = ImageChops.difference(first, second).convert("L")
    histogram = diff.histogram()
    changed = sum(histogram[16:])
    return {"same_size": True, "size": first.size, "max_channel_diff": max(i for i, n in enumerate(histogram) if n) if
            any(histogram) else 0, "fraction_pixels_diff_gt16": changed / (first.size[0] * first.size[1])}


# ------------------------------------------------------------------ hardcode lint
ABSOLUTE = re.compile(r"(/Users/|/home/|/scratch\d*/|/data/pclancy|~/)")


def lint_script(path: Path, min_array: int = 6) -> list[str]:
    """Absolute paths in code, and literal numeric arrays that look like data, not layout."""
    source = path.read_text()
    findings = []
    tree = ast.parse(source)
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if ABSOLUTE.search(node.value):
                findings.append(f"{rel(path)}:{node.lineno}: absolute path {node.value[:80]!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, float)]
            if len(values) >= min_array and len({round(v, 6) for v in values}) == len(values):
                if not _is_layout(values):
                    findings.append(f"{rel(path)}:{node.lineno}: literal float array of {len(values)} values")
    return findings


def _is_layout(values: list[float]) -> bool:
    """Evenly spaced ticks, or fractions in [0, 1] with few decimals, are layout rather than data."""
    steps = {round(b - a, 6) for a, b in zip(values, values[1:])}
    few_decimals = all(len(repr(v).split(".")[-1]) <= 3 for v in values)
    return len(steps) == 1 or few_decimals


# ------------------------------------------------------------------ evidence
class Evidence:
    """Collects what a generator read, wrote and checked; write() stores it as JSON."""

    def __init__(self, script: str, name: str):
        self.script = Path(script).resolve()
        self.item_dir = next(p for p in self.script.parents if (p / "PROVENANCE.md").exists())
        self.name = name
        self.out_dir = self.item_dir / "rerun_output"
        self.out_dir.mkdir(exist_ok=True)
        self.inputs: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.checks: list[dict] = []

    def input(self, repo_path: str) -> Path:
        path = REPO / repo_path
        if not path.exists():
            raise FileNotFoundError(f"input missing: {repo_path}")
        self.inputs[repo_path] = sha256(path) if path.is_file() else "directory"
        return path

    def output(self, filename: str) -> Path:
        return self.out_dir / filename

    def check(self, name: str, passed: bool, details) -> bool:
        self.checks.append({"check": name, "passed": bool(passed), "details": details})
        return passed

    def compare_table(self, rows: list[list[str]], tex_name: str, header_rows: int = 1) -> bool:
        target = self.output(re.sub(r"[^\w-]", "_", self.name) + ".tex")
        target.write_text(render_table(rows))
        self.outputs[rel(target)] = sha256(target)
        submitted = submitted_table(tex_name, self.name, header_rows)
        errors = compare_rows(submitted, rows)
        return self.check(f"values equal submitted {tex_name} {self.name} at displayed precision",
                          not errors, errors or f"{len(rows)} rows match")

    def compare_pdf(self, produced: Path, reference: str, dpi: int = 100, tolerance: float = 1e-3) -> bool:
        reference_path = REPO / reference
        self.outputs[rel(produced)] = sha256(produced)
        if sha256(produced) == sha256(reference_path):
            return self.check(f"identical to {reference}", True, "sha256 equal")
        metrics = image_difference(rasterize(produced, dpi), rasterize(reference_path, dpi))
        passed = metrics.get("same_size") and metrics["fraction_pixels_diff_gt16"] <= tolerance
        return self.check(f"renders like {reference} at {dpi} dpi", passed, metrics)

    def write(self) -> int:
        passed = all(c["passed"] for c in self.checks) and bool(self.checks)
        record = {
            "item_dir": rel(self.item_dir), "name": self.name, "date": date.today().isoformat(),
            "command": f"python {rel(self.script)}", "script_sha256": sha256(self.script),
            "python": sys.version.split()[0], "inputs": self.inputs, "outputs": self.outputs,
            "hardcode_lint": lint_script(self.script), "checks": self.checks,
            "result": "PASS" if passed else "FAIL",
        }
        folder = self.item_dir / "evidence"
        folder.mkdir(exist_ok=True)
        target = folder / (re.sub(r"[^\w-]", "_", self.name) + ".json")
        target.write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(f"{record['result']} {self.name}: {len(self.checks)} checks -> {rel(target)}")
        for check in self.checks:
            if not check["passed"]:
                print(f"  FAIL {check['check']}: {check['details']}")
        return 0 if passed else 1


def repo_tools_path() -> Path:
    return PACKAGE / "tools"


if shutil.which("pdftoppm") is None:  # pragma: no cover - environment note only
    print("note: pdftoppm not found; figure render comparison unavailable", file=sys.stderr)

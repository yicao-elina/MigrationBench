#!/usr/bin/env python3
"""Prepare a clean PW-only rerun directory on Rockfish; never submit a job."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


SCRATCH_ROOT = Path("/scratch16/pclancy3/yi")
QE_ROOT = Path("/data/apps/extern/qe/7.2/bin")
PW_BINARY_SHA256 = "e893e4835199422ac09218709bab451b6d47620fdfaa873d135c582da028cb81"
PP_BINARY_SHA256 = "ab617735f7d8ea62870a2886a8afcfab4fd342a6a56221dec63c8036dbc6c328"
RUNS = {
    "base_spin": {
        "source": Path("/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap/spin"),
        "input_sha256": "d16a1cc8b78132fdadea7c7433a2f1a34c160e3a03ad93a33245c5b8944f1ef0",
        "scheduler_sha256": "93842b62517e6965f76cfb152dff6ec2882efc540a7a0978d65024bf556e8cb9",
        "pp_input_sha256": "4b68df9d1eb6d17570dc6984770fe55a75b47bed251c3f097f6d31a20ed662e1",
        "pseudo_dir": Path("/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE"),
    },
    "mag3": {
        "source": Path("/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap/spin/mag3"),
        "input_sha256": "c137554c50a31a4364f86d9404f847f290120dbffeb15533d9569353aded99d7",
        "scheduler_sha256": "93842b62517e6965f76cfb152dff6ec2882efc540a7a0978d65024bf556e8cb9",
        "pp_input_sha256": "4b68df9d1eb6d17570dc6984770fe55a75b47bed251c3f097f6d31a20ed662e1",
        "pseudo_dir": Path("/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE"),
    },
    "charged3_spin": {
        "source": Path("/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap/charged_3/spin"),
        "input_sha256": "00dd50b361ab3acf5157b6c9146ebd968485fa3980133ecb269232cde0b41120",
        "scheduler_sha256": "93842b62517e6965f76cfb152dff6ec2882efc540a7a0978d65024bf556e8cb9",
        "pp_input_sha256": "f60fef393b019b56ca865c9f1674018e03e176a6d61742d382556a4588a480b6",
        "pseudo_dir": Path("/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE"),
    },
}
PSEUDOPOTENTIAL_SHA256 = {
    "Sb_PBE_FR.SG15v1.2.UPF": "998415c8a49d852c7376645c0a896110258aa7bb40b5c331a1e2874ed34e8132",
    "Te_PBE_FR.SG15v1.2.UPF": "1b6d3177b82988cecabebf7da1636825613b7270bc269795a1fb6c12743e12fa",
    "Cr_PBE_FR.SG15v1.2.UPF": "a9352047039238c12614fb6c3ed93acb6e84b2cd9011db2f62eda1858045c5e4",
}
PW_COMMAND = "mpirun -np 4 /data/apps/extern/qe/7.2/bin/pw.x < relax.fresh.in > relax.fresh.out"
PP_COMMAND = "mpirun -np 1 /data/apps/extern/qe/7.2/bin/pp.x < pp.in > pp.out"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact_file(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise SystemExit("source is missing or not a regular file: " + str(path))
    actual = sha256(path)
    if actual != expected_sha256:
        raise SystemExit("source hash changed: %s expected=%s actual=%s" % (path, expected_sha256, actual))


def fresh_input(source_text: str) -> str:
    control = re.search(r"(?ims)^\s*&CONTROL\b(.*?^\s*/)", source_text)
    if not control:
        raise SystemExit("input has no parseable &CONTROL namelist")
    block = control.group(1)
    block, outdir_count = re.subn(
        r"(?im)^([ \t]*outdir[ \t]*=[ \t]*)['\"][^'\"]*['\"][ \t]*,?[ \t]*$",
        r"\g<1>'./out_fresh/'",
        block,
    )
    if outdir_count != 1:
        raise SystemExit("expected one outdir assignment in &CONTROL")
    restart_re = re.compile(r"(?im)^\s*restart_mode\s*=")
    if restart_re.search(block):
        block, count = re.subn(
            r"(?im)^([ \t]*restart_mode[ \t]*=[ \t]*)['\"][^'\"]*['\"][ \t]*,?[ \t]*$",
            r"\g<1>'from_scratch'",
            block,
        )
        if count != 1:
            raise SystemExit("restart_mode exists but could not be replaced safely")
    else:
        block, count = re.subn(
            r"(?im)^([ \t]*outdir[ \t]*=[ \t]*['\"][^'\"]*['\"][ \t]*,?[ \t]*)$",
            r"\1\n  restart_mode = 'from_scratch'",
            block,
        )
        if count != 1:
            raise SystemExit("could not insert explicit restart_mode after outdir")
    return source_text[: control.start(1)] + block + source_text[control.end(1) :]


def fresh_scheduler(source_text: str, run_name: str) -> str:
    if source_text.count(PW_COMMAND.replace("relax.fresh.in", "relax.in").replace("relax.fresh.out", "relax.out")) != 1:
        raise SystemExit("source scheduler does not contain exactly one expected pw.x command")
    if source_text.count(PP_COMMAND) != 1:
        raise SystemExit("source scheduler does not contain exactly one known pp.x command")
    text = source_text.replace(
        PW_COMMAND.replace("relax.fresh.in", "relax.in").replace("relax.fresh.out", "relax.out"),
        PW_COMMAND,
    )
    text = text.replace(PP_COMMAND, "# PP omitted: historical pp.out fails with wrong nx; diagnose separately.")
    text, job_count = re.subn(r'(?m)^#SBATCH --job-name=.*$', '#SBATCH --job-name="qe-' + run_name + '-pw-fresh"', text)
    if job_count != 1:
        raise SystemExit("expected one Slurm job-name directive")
    module_anchor = "ml qe/7.2\n"
    if text.count(module_anchor) != 1:
        raise SystemExit("expected one QE 7.2 module load")
    text = text.replace(module_anchor, module_anchor + "set -e\n", 1)
    text = text.replace('echo "Running pp.x for post-processing"\n', "")
    pw_line = PW_COMMAND
    if text.count(pw_line) != 1:
        raise SystemExit("fresh pw.x command did not survive scheduler rewrite")
    text = text.replace(pw_line, "mkdir -p out_fresh\n" + pw_line, 1)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=sorted(RUNS))
    args = parser.parse_args()
    spec = RUNS[args.run]
    source = spec["source"]
    input_path = source / "relax.in"
    scheduler_path = source / "sbatch-relax.slurm"
    pp_input_path = source / "pp.in"
    require_exact_file(input_path, spec["input_sha256"])
    require_exact_file(scheduler_path, spec["scheduler_sha256"])
    require_exact_file(pp_input_path, spec["pp_input_sha256"])
    for pseudo_name, expected in PSEUDOPOTENTIAL_SHA256.items():
        require_exact_file(spec["pseudo_dir"] / pseudo_name, expected)
    pw_binary = QE_ROOT / "pw.x"
    if pw_binary.is_symlink() or not pw_binary.is_file() or not os.access(str(pw_binary), os.X_OK):
        raise SystemExit("QE 7.2 pw.x is not present/executable: " + str(pw_binary))
    actual_pw_sha = sha256(pw_binary)
    if actual_pw_sha != PW_BINARY_SHA256:
        raise SystemExit("QE 7.2 pw.x identity changed: expected=%s actual=%s" % (PW_BINARY_SHA256, actual_pw_sha))
    if not SCRATCH_ROOT.is_dir() or not os.access(str(SCRATCH_ROOT), os.W_OK | os.X_OK):
        raise SystemExit("scratch work root is unavailable or not writable: " + str(SCRATCH_ROOT))

    run_dir = Path(tempfile.mkdtemp(prefix="qe_" + args.run + "_pw_fresh_", dir=str(SCRATCH_ROOT)))
    shutil.copy2(str(input_path), str(run_dir / "relax.source.in"))
    shutil.copy2(str(scheduler_path), str(run_dir / "sbatch-relax.source.slurm"))
    shutil.copy2(str(pp_input_path), str(run_dir / "pp.reference.in"))
    (run_dir / "relax.fresh.in").write_text(fresh_input(input_path.read_text(encoding="utf-8")), encoding="utf-8")
    (run_dir / "sbatch-relax-pw-fresh.slurm").write_text(
        fresh_scheduler(scheduler_path.read_text(encoding="utf-8"), args.run), encoding="utf-8"
    )
    (run_dir / "out_fresh").mkdir()
    manifest = {
        "run": args.run,
        "source_directory": str(source),
        "working_directory": str(run_dir),
        "source_input_sha256": sha256(input_path),
        "source_scheduler_sha256": sha256(scheduler_path),
        "source_pp_input_sha256": sha256(pp_input_path),
        "fresh_input_sha256": sha256(run_dir / "relax.fresh.in"),
        "fresh_scheduler_sha256": sha256(run_dir / "sbatch-relax-pw-fresh.slurm"),
        "pseudopotentials": {name: sha256(spec["pseudo_dir"] / name) for name in PSEUDOPOTENTIAL_SHA256},
        "pw_binary": str(pw_binary),
        "pw_binary_sha256": actual_pw_sha,
        "pp_binary": str(QE_ROOT / "pp.x"),
        "pp_binary_sha256": PP_BINARY_SHA256,
        "restart_mode": "from_scratch",
        "outdir": "./out_fresh/",
        "postprocessing": "not run; historical pp.out is retained as wrong-nx failure evidence",
        "job_submitted": False,
    }
    (run_dir / "fresh_start_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Prepared clean PW-only rerun directory: " + str(run_dir))
    print("Review files, then submit manually with: cd %s && sbatch sbatch-relax-pw-fresh.slurm" % run_dir)
    print("No job was submitted by this helper.")


if __name__ == "__main__":
    main()

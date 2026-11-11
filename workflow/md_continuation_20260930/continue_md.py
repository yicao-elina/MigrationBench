#!/usr/bin/env python
import argparse, json, os, signal, tempfile
from pathlib import Path

from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator

STOP = False

def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f"received signal {signum}; stopping at next checkpoint", flush=True)

def atomic_json(path, payload):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--start-step", type=int, required=True)
    ap.add_argument("--target-step", type=int, default=1_000_000)
    ap.add_argument("--temperature", type=float, default=600.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    status_path = run / "status.json"
    restart_path = run / "latest_restart.extxyz"
    segment_path = run / f"segment_from_{args.start_step:09d}.extxyz"
    manifest = run / "run_manifest.json"

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    if status_path.exists():
        status = json.loads(status_path.read_text())
        step = int(status["last_step"])
        if not restart_path.exists():
            raise RuntimeError("status.json exists but latest_restart.extxyz is missing")
        atoms = read(restart_path, index=-1)
        if not atoms.has("momenta"):
            raise RuntimeError("checkpoint has no momenta")
        print(f"resuming from checkpoint step {step}", flush=True)
    else:
        step = args.start_step
        atoms = read(args.input, index=-1)
        if not atoms.has("momenta"):
            raise RuntimeError("input final frame has no momenta")
        if segment_path.exists():
            raise RuntimeError(f"refusing to overwrite existing {segment_path}")
        write(restart_path, atoms, format="extxyz")
        status = {"last_step": step, "target_step": args.target_step, "source": args.input}
        atomic_json(status_path, status)

    if len(atoms) != 2050 or not all(atoms.get_pbc()):
        raise RuntimeError(f"unexpected structure: natoms={len(atoms)} pbc={atoms.get_pbc()}")
    if step >= args.target_step:
        print("already complete", flush=True)
        return

    calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float64")
    atoms.calc = calc
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=args.temperature, friction=0.01)

    write_every = 200
    def checkpoint():
        nonlocal step
        write(restart_path, atoms, format="extxyz")
        atomic_json(status_path, {"last_step": step, "target_step": args.target_step,
                                  "model": args.model, "segment": str(segment_path),
                                  "natoms": len(atoms), "pbc": atoms.get_pbc().tolist()})

    def save_frame():
        nonlocal step
        step += write_every
        atoms.info["step"] = step
        atoms.info["time_fs"] = float(step)
        write(segment_path, atoms, format="extxyz", append=True)
        checkpoint()
        if STOP:
            raise SystemExit(99)

    dyn.attach(save_frame, interval=write_every)
    atomic_json(manifest, {"input": args.input, "model": args.model, "start_step": args.start_step,
                           "target_step": args.target_step, "dt_fs": 1.0,
                           "write_interval_steps": write_every, "temperature_K": args.temperature,
                           "friction": 0.01, "segment": str(segment_path)})
    remaining = args.target_step - step
    dyn.run(remaining)
    # Attach callback fires at exact 200-step boundaries; make final state authoritative.
    if step < args.target_step:
        step = args.target_step
    checkpoint()
    print(f"completed step {step}", flush=True)

if __name__ == "__main__":
    main()

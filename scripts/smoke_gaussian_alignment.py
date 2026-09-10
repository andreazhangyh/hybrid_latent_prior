"""Run four independent 108-update smoke checks after the T0 gates pass."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[2, 3, 4, 5])
    args = parser.parse_args()
    if len(set(args.gpus)) != 4:
        raise ValueError("Each independent experiment needs a distinct GPU")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise RuntimeError("Smoke manifest already exists; inspect its PIDs/results before restarting")
    manifest = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(), runs={})
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=str(ROOT))
    (args.output / "worktree.patch").write_bytes(patch)
    paths = list((ROOT / "tests").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    paths += list((ROOT / "isaacgymenvs/learning").glob("cvae*.py"))
    paths += list((ROOT / "isaacgymenvs/cfg/train/imitation").glob("HybridDistill*.yaml"))
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    processes = []
    for experiment, gpu in zip(("A0", "A1", "A2", "A3"), args.gpus):
        output = args.output / experiment
        output.mkdir()
        command = [sys.executable, "-u", str(ROOT / "scripts/check_alignment_training.py"),
                   "--epochs", "9", "--audit-gradients", "--output", str(output), "--",
                   "headless=True", "task=LafanImitation", "train=imitation/HybridDistill" + experiment,
                   "experiment=alignment_" + experiment.lower() + "_smoke",
                   "expert=pretrained_weights/imitation/imitation_expert"]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_JIT="0", HYDRA_FULL_ERROR="1",
                   LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1] / "lib") + ":" + env.get("LD_LIBRARY_PATH", ""))
        with (output / "train.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=str(ROOT / "isaacgymenvs"), env=env,
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        processes.append((experiment, process))
        manifest["runs"][experiment] = dict(pid=process.pid, gpu=gpu, command=command)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["runs"], indent=2), flush=True)
    results = {}
    for experiment, process in processes:
        code = process.wait()
        results[experiment] = dict(exit_code=code)
        if code == 0:
            rows = [json.loads(line) for line in (args.output / experiment / "epochs.jsonl").read_text().splitlines()]
            results[experiment].update(last_epoch=rows[-1]["epoch"], updates=rows[-1]["optimizer_updates"],
                                       run_directory=rows[-1]["run_directory"])
            assert 100 <= rows[-1]["optimizer_updates"] <= 200
        print(experiment, results[experiment], flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(row["exit_code"] for row in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

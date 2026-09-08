"""Launch one independent A1 run with the original A0 seed and hyperparameters."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    assert audit["exact_forward_and_rvq_state_equality"]
    assert audit["all_values_and_gradients_finite"]
    smoke = [json.loads(line) for line in args.smoke.read_text().splitlines()]
    assert len(smoke) >= 3 and smoke[-1]["epoch"] >= 3
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "launch.json"
    if metadata_path.exists():
        raise RuntimeError("Launch record already exists; inspect its PID before considering another run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    assert all(row["direction"] == "post_to_prior" and row["commit"] == commit for row in smoke)
    subprocess.check_call(["git", "diff", "--exit-code", "HEAD", "--",
                           "isaacgymenvs/learning/cvae_network_builder.py",
                           "isaacgymenvs/learning/cvae_model.py",
                           "isaacgymenvs/learning/cvae_agent.py",
                           "isaacgymenvs/cfg/train/imitation/HybridDistill.yaml"], cwd=str(ROOT))
    command = [sys.executable, "-u", "train.py", "headless=True", "task=LafanImitation",
               "train=imitation/HybridDistill", "experiment=alignment_a1_seed42",
               "expert=pretrained_weights/imitation/imitation_expert",
               "train.params.config.latent_align.direction=post_to_prior"]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTORCH_JIT="0", HYDRA_FULL_ERROR="1",
                       LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1] / "lib")
                       + ":" + environment.get("LD_LIBRARY_PATH", ""))
    metadata = dict(commit=commit, command=command, cwd=str(ROOT / "isaacgymenvs"), gpu=args.gpu,
                    seed=42, initialization="from scratch, same expert as A0",
                    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    audit=str(args.audit.resolve()), smoke=str(args.smoke.resolve()))
    with (args.output / "train.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=metadata["cwd"], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    metadata["pid"] = process.pid
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""Bounded training check without changing max_epochs or training hyperparameters.

Run from isaacgymenvs with the usual train.py Hydra arguments after --.
The first real distillation batch is copied for offline numerical validation.
"""
import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

import isaacgym  # Must precede torch with Isaac Gym Preview 4.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaacgymenvs.learning.cvae_agent import cVAEAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    original_calc = cVAEAgent.calc_gradients_non_rl
    original_log = cVAEAgent._log_train_info
    captured = [False]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()

    def calc(agent, batch):
        if not captured[0]:
            def capture(model, inputs):
                data = {key: value[:256].detach().cpu().clone()
                        if torch.is_tensor(value) else value
                        for key, value in inputs[0].items()}
                data["expert_mu"] = batch["expert_mu"][:256].detach().cpu().clone()
                torch.save(data, str(args.output / "validation_batch.pt"))
                captured[0] = True
                handle.remove()
            handle = agent.model.register_forward_pre_hook(capture)
        return original_calc(agent, batch)

    def log(agent, info, frame):
        original_log(agent, info, frame)
        required = ["latent_align_loss", "latent_align_post_norm", "latent_align_prior_norm",
                    "latent_align_residual_norm", "latent_align_code_entropy_layer_7"]
        for key in required:
            assert key in info, "Missing training metric: " + key
        scalars = {}
        for key, values in info.items():
            if isinstance(values, list) and values and torch.is_tensor(values[0]):
                value = torch.stack([x.detach().float().mean() for x in values]).mean()
                assert torch.isfinite(value), "Nonfinite training metric: " + key
                scalars[key] = value.item()
        row = dict(epoch=agent.epoch_num, frame=frame, metrics=scalars,
                   commit=commit, direction=agent._latent_align_config["direction"],
                   alignment_coef=agent.vae_kl_loss_coef,
                   run_directory=str(Path(agent.network_path).parent.resolve()))
        with (args.output / "epochs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        agent.writer.flush()
        print("ALIGNMENT_CHECK", json.dumps(row), flush=True)
        if agent.epoch_num >= args.epochs:
            agent.save(str(args.output / "check_checkpoint"))
            agent.writer.close()
            raise SystemExit(0)

    cVAEAgent.calc_gradients_non_rl = calc
    cVAEAgent._log_train_info = log
    sys.argv = [str(ROOT / "isaacgymenvs/train.py")] + [x for x in overrides if x != "--"]
    runpy.run_path(str(ROOT / "isaacgymenvs/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()

"""Bounded training check without changing max_epochs or training hyperparameters.

Run from isaacgymenvs with the usual train.py Hydra arguments after --.
The first real distillation batch is copied for offline numerical validation.
"""
import argparse
import json
import math
import os
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
    parser.add_argument("--audit-gradients", action="store_true")
    parser.add_argument("--review-epochs", default="")
    parser.add_argument("--review-num-envs", type=int, default=10)
    parser.add_argument("--review-steps", type=int, default=150)
    parser.add_argument("--review-modes", default="prior,posterior")
    args, overrides = parser.parse_known_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    review_epochs = {int(x) for x in args.review_epochs.split(",") if x}
    review_modes = [x for x in args.review_modes.split(",") if x]
    train_override = next((x.split("=", 1)[1] for x in overrides if x.startswith("train=")), None)
    task_override = next((x.split("=", 1)[1] for x in overrides if x.startswith("task=")), "LafanImitation")
    original_calc = cVAEAgent.calc_gradients_non_rl
    original_log = cVAEAgent._log_train_info
    captured = [False]
    gradient_epoch = [-1]
    norm_history = []
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()

    def calc(agent, batch):
        if args.audit_gradients and gradient_epoch[0] != agent.epoch_num:
            def audit(model, inputs, output):
                enc = model.a2c_network.encoder
                groups = {name: list(getattr(enc, name).parameters()) for name in
                          ("_post_net", "_post_loc_net", "_prior_net", "_prior_loc_net", "_prior_logstd_net")
                          if hasattr(enc, name)}
                params = [p for group in groups.values() for p in group]
                losses = dict(action=(output["mus"] - batch["expert_mu"]).square().sum(-1).mean(),
                              alignment=output["vae_kl_loss"], commitment=output["vae_commit_loss"])
                weights = dict(action=agent.expert_loss_coef, alignment=agent.vae_kl_loss_coef,
                               commitment=agent.vae_commit_loss_coef)
                records = {}
                for name, loss in losses.items():
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    assert all(torch.isfinite(g).all() for g in grads if g is not None)
                    offset, norms = 0, {}
                    for group, parameters in groups.items():
                        selected = grads[offset:offset + len(parameters)]
                        norms[group] = math.sqrt(sum(g.float().square().sum().item() for g in selected if g is not None))
                        offset += len(parameters)
                    records[name] = dict(raw_l2=norms, weight=weights[name],
                                         weighted_l2={key: value * weights[name] for key, value in norms.items()})
                ratio = math.sqrt(sum(x*x for x in records["alignment"]["weighted_l2"].values())) / max(
                    math.sqrt(sum(x*x for x in records["action"]["weighted_l2"].values())), 1e-12)
                assert ratio < 100, "Weighted alignment gradient exceeds action by 100x"
                with (args.output / "gradients.jsonl").open("a") as f:
                    f.write(json.dumps(dict(epoch=agent.epoch_num, losses=records,
                                            weighted_alignment_action_ratio=ratio)) + "\n")
                gradient_epoch[0] = agent.epoch_num
                audit_handle.remove()
            audit_handle = agent.model.register_forward_hook(audit)
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
                   optimizer_updates=agent.epoch_num * len(agent.dataset) * agent.mini_epochs_num,
                   alignment_config=dict(agent._latent_align_config),
                   commit=commit, direction=agent._latent_align_config["direction"],
                   alignment_coef=agent.vae_kl_loss_coef,
                   run_directory=str(Path(agent.network_path).parent.resolve()))
        if args.audit_gradients:
            for key in ("latent_align_logstd_lower_fraction", "latent_align_logstd_upper_fraction"):
                assert scalars.get(key, 0) < 0.5, "At least half of prior logstd is clamped"
            norm_history.append(max(scalars["latent_align_post_norm"], scalars["latent_align_prior_norm"]))
            assert not (len(norm_history) >= 3 and all(x > 10 * norm_history[0] for x in norm_history[-3:])), "Sustained latent norm explosion"
        with (args.output / "epochs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        agent.writer.flush()
        print("ALIGNMENT_CHECK", json.dumps(row), flush=True)
        if agent.epoch_num in review_epochs:
            assert train_override is not None, "Milestone review requires a train=... override"
            review_dir = args.output / "reviews" / ("epoch_%d" % agent.epoch_num)
            review_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_base = review_dir / "checkpoint"
            agent.save(str(checkpoint_base))
            checkpoint_file = str(checkpoint_base) + ".pth"
            for mode in review_modes:
                review_output = review_dir / ("%s_rollout.json" % mode)
                command = [
                    sys.executable, "-u", str(ROOT / "scripts/evaluate_alignment_rollout.py"),
                    "--output", str(review_output),
                    "--mode", mode,
                    "--steps", str(args.review_steps),
                    "--",
                    "test=True",
                    "headless=True",
                    "num_envs=%d" % args.review_num_envs,
                    "task=%s" % task_override,
                    "train=%s" % train_override,
                    "checkpoint=%s" % checkpoint_file,
                ]
                completed = subprocess.run(
                    command,
                    cwd=str(ROOT / "isaacgymenvs"),
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                log_path = review_dir / ("%s_rollout.log" % mode)
                log_path.write_text(completed.stdout)
                assert completed.returncode == 0, "Milestone rollout failed: %s" % log_path
                json.loads(review_output.read_text())
                with (args.output / "reviews.jsonl").open("a") as f:
                    f.write(json.dumps(dict(epoch=agent.epoch_num, mode=mode, output=str(review_output))) + "\n")
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

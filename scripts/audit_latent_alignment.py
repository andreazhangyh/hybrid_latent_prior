"""Offline Phase 0/1 audit on a saved real model-input batch (no Isaac Gym)."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import types
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_latent_alignment_rvq import make_model, gradients, legacy_forward, seed


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed()
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    batch = torch.load(str(args.batch), map_location="cpu")
    target = batch.pop("expert_mu")
    original = make_model(full=True)
    original.load_state_dict(checkpoint["model"], strict=True)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                  checkpoint_epoch=checkpoint["epoch"], batch=str(args.batch),
                  batch_sha256=digest(args.batch), batch_size=batch["obs"].shape[0],
                  batch_description="Real first-update model inputs; goal_obs already normalized at capture",
                  python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(),
                  strict_checkpoint_load=True, seed=2026, results={})
    reference = None
    reference_state = None
    for direction in ("legacy", "default", "bidirectional", "post_to_prior"):
        model = copy.deepcopy(original)
        enc = model.a2c_network.encoder
        if direction == "legacy":
            enc.forward = types.MethodType(legacy_forward(), enc)
        elif direction != "default":
            enc._latent_align_config["direction"] = direction
        # Freeze input normalizers at checkpoint values, retain real RVQ training behavior.
        model.eval()
        enc.train()
        seed(2026)
        result = model(dict(batch))
        action = (result["mus"] - target).pow(2).sum(-1).mean()
        losses = dict(align=result["vae_kl_loss"], action=action, commit=result["vae_commit_loss"])
        losses["full"] = 10 * action + 0.1 * losses["align"] + losses["commit"]
        for key, value in result.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), key
        keys = ("mus", "post_mu", "prior_mu", "vae_kl_loss", "vae_commit_loss")
        if reference is None:
            reference = {key: result[key].detach().clone() for key in keys}
            reference_state = copy.deepcopy(model.state_dict())
        else:
            for key in keys:
                assert torch.equal(reference[key], result[key]), (direction, key)
            for key, value in model.state_dict().items():
                assert torch.equal(reference_state[key], value), (direction, key)
        entry = dict(losses={key: value.item() for key, value in losses.items()}, gradients={},
                     metrics={key: value.mean().item() for key, value in result.items()
                              if key.startswith("latent_align_")})
        for kind, loss in losses.items():
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            actual = gradients(model)
            for name, value in actual.items():
                expected = (kind in ("align", "full") if "prior" in name
                            else not (kind == "align" and direction == "post_to_prior"))
                assert value > 0 if expected else value == 0, (direction, kind, name, value)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()
            entry["gradients"][kind] = actual
        report["results"][direction] = entry
    report["exact_forward_and_rvq_state_equality"] = True
    report["all_values_and_gradients_finite"] = True
    original.eval()
    with torch.no_grad():
        evaluated = original(dict(batch))
    report["a0_eval_all_layers_metrics"] = {key: value.mean().item()
                                            for key, value in evaluated.items()
                                            if key.startswith("latent_align_")}
    for name in ("config.yaml", "net_config.yaml", "hash_code.txt"):
        shutil.copyfile(str(args.baseline_run / name), str(args.output / ("a0_" + name)))
    expert = ROOT / "isaacgymenvs/pretrained_weights/imitation/imitation_expert/nn/imitation_expert_weight.pth"
    report["expert_sha256"] = digest(expert)
    motion_root = ROOT / "assets/amp/motions/LAFAN_ALL"
    report["motion_sha256"] = {str(path.resolve()): digest(path)
                               for path in sorted(motion_root.rglob("*.npy"))}
    assert report["motion_sha256"], "No baseline motion data found"
    (args.output / "user_setup.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "--", "isaacgymenvs/tasks/amp/poselib"], cwd=str(ROOT)))
    events = EventAccumulator(str(args.baseline_run / "summaries"), size_guidance={"scalars": 0})
    events.Reload()
    curves = {}
    with (args.output / "baseline_scalars.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "wall_time", "value"])
        for tag in events.Tags()["scalars"]:
            values = events.Scalars(tag)
            for value in values:
                writer.writerow([tag, value.step, value.wall_time, value.value])
            if "loss" in tag:
                curves[tag] = values
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(curves), 1, figsize=(10, max(3, len(curves) * 2.5)), squeeze=False)
    for ax, (tag, values) in zip(axes[:, 0], curves.items()):
        ax.plot([x.step for x in values], [x.value for x in values], linewidth=0.7)
        ax.set_title(tag)
        ax.set_xlabel("Environment frames")
    fig.tight_layout()
    fig.savefig(str(args.output / "baseline_losses.png"), dpi=120)
    plt.close(fig)
    report["baseline_last_scalars"] = {tag: dict(step=values[-1].step, value=values[-1].value)
                                       for tag, values in curves.items() if values}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], universal_newlines=True))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Verify completed T1 evidence, including actual learned checkpoint roundtrips."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_gaussian_alignment import config
from tests.test_latent_alignment_rvq import make_model, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = json.loads((args.output / "results.json").read_text())
    summary = {}
    for experiment, result in results.items():
        assert result["exit_code"] == 0 and result["updates"] == 108
        directory = args.output / experiment
        rows = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
        gradients = [json.loads(line) for line in (directory / "gradients.jsonl").read_text().splitlines()]
        assert len(rows) == len(gradients) == 9
        for row in gradients:
            for kind, loss in row["losses"].items():
                for name, norm in loss["raw_l2"].items():
                    expected = (kind == "alignment" if "prior" in name else
                                not (kind == "alignment" and experiment in ("A1", "A3")))
                    assert norm > 0 if expected else norm == 0, (experiment, row["epoch"], kind, name, norm)
        events = EventAccumulator(str(Path(result["run_directory"]) / "summaries"), size_guidance={"scalars": 0})
        events.Reload()
        tags = events.Tags()["scalars"]
        for tag in tags:
            assert all(math.isfinite(value.value) for value in events.Scalars(tag))
        for key in rows[-1]["metrics"]:
            if key.startswith("latent_align_"):
                tag = "latent_align/" + key[len("latent_align_"):]
                assert len(events.Scalars(tag)) == 9, (experiment, tag)
        for tag in ("latent_align/coef", "latent_align/weighted_loss", "latent_align/posterior_std"):
            assert len(events.Scalars(tag)) == 9
        base = yaml.safe_load((ROOT / "isaacgymenvs/runs/imitation_hybrid_b6offnv3/config.yaml").read_text())
        actual = yaml.safe_load((Path(result["run_directory"]) / "config.yaml").read_text())
        for cfg in (base, actual):
            cfg.pop("experiment")
            for key in ("name", "full_experiment_name"):
                cfg["train"]["params"]["config"].pop(key)
        alignment = actual["train"]["params"]["config"].pop("latent_align")
        for key, value in config(experiment).items():
            assert alignment[key] == value
        base_coef = base["train"]["params"]["config"].pop("vae_kl_loss_coef")
        actual_coef = actual["train"]["params"]["config"].pop("vae_kl_loss_coef")
        assert actual_coef == (base_coef.replace("0.1", "0.018") if experiment in ("A2", "A3") else base_coef)
        assert base == actual, "Unexpected training configuration difference: " + experiment
        state = torch.load(str(directory / "check_checkpoint.pth"), map_location="cpu")["model"]
        models = [make_model(full=True, alignment=config(experiment)) for _ in range(2)]
        for model in models:
            model.load_state_dict(state, strict=True)
            model.eval()
        if experiment in ("A2", "A3"):
            assert models[0].a2c_network.encoder._prior_logstd_net.weight.abs().sum() > 0
        batch = torch.load(str(directory / "validation_batch.pt"), map_location="cpu")
        batch.pop("expert_mu")
        with torch.no_grad():
            outputs = [model(dict(batch)) for model in models]
        for key in ("mus", "vae_kl_loss", "vae_commit_loss"):
            assert torch.equal(outputs[0][key], outputs[1][key]), (experiment, key)
            assert torch.isfinite(outputs[0][key]).all()
        summary[experiment] = dict(updates=108, tensorboard_finite=True, gradient_matrix_passed=True,
                                   trained_checkpoint_roundtrip=True, configuration_verified=True,
                                   max_weighted_alignment_action_ratio=max(x["weighted_alignment_action_ratio"] for x in gradients),
                                   first_alignment_grad_l2=gradients[0]["losses"]["alignment"]["raw_l2"],
                                   last_metrics=rows[-1]["metrics"], run_directory=result["run_directory"])
    (args.output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

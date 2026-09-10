"""Real RVQ/decoder tests; no Isaac Gym import or simulator required."""
import copy
import random
import subprocess
import types
import unittest
from pathlib import Path

import torch
import yaml

from isaacgymenvs.learning.cvae_amp_network_builder import cVAEAMPBuilder
from isaacgymenvs.learning.cvae_amp_model import ModelcVAEAMPContinuous
from isaacgymenvs.learning.cvae_network_builder import ContinuousEncoder

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "2113e72461f254060be72b3b920c888b22afcf8c"


def seed(value=42):
    random.seed(value)
    torch.manual_seed(value)


def make_model(direction=None, full=False, alignment=None):
    with (ROOT / "isaacgymenvs/cfg/train/imitation/HybridDistill.yaml").open() as f:
        params = yaml.safe_load(f)["params"]["network"]
    if not full:
        for name in ("enc", "dec", "value", "disc"):
            params[name]["units"] = [32, 32]
    builder = cVAEAMPBuilder()
    builder.load(params)
    config = dict(input_shape=(105,), goal_input_shape=(225,), latent_shape=(64,),
                  actions_num=28, no_goal=False, enc_type="hybrid", enc_scale=0.3,
                  continuous_enc_style="quantcond", code_num=1024 if full else 16,
                  quant_type="rvq", num_quants=8, amp_input_shape=(210,))
    if direction is not None:
        config["latent_align"] = dict(direction=direction)
    if alignment is not None:
        config["latent_align"] = alignment
    net = builder.build("cvae_amp", **config)
    return ModelcVAEAMPContinuous.Network(net, obs_shape=(105,),
                                        normalize_input=full, normalize_value=full,
                                        value_size=1, enc_type="hybrid")


def groups(model):
    enc = model.a2c_network.encoder
    return {name: list(getattr(enc, name).parameters()) for name in
            ("_post_net", "_post_loc_net", "_prior_net", "_prior_loc_net")}


def gradients(model):
    return {name: sum(p.grad.abs().sum().item() for p in params if p.grad is not None)
            for name, params in groups(model).items()}


def legacy_forward():
    source = subprocess.check_output([
        "git", "show", BASE_COMMIT + ":isaacgymenvs/learning/cvae_network_builder.py"
    ], cwd=str(ROOT), universal_newlines=True)
    namespace = {"__name__": "legacy_cvae_network_builder"}
    exec(compile(source, "legacy_cvae_network_builder.py", "exec"), namespace)
    return namespace["ContinuousEncoder"].forward


class RealRVQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.legacy = staticmethod(legacy_forward())

    def setUp(self):
        seed()
        self.model = make_model()
        self.batch = dict(obs=torch.randn(32, 105), goal_obs=torch.randn(32, 225),
                          prev_actions=torch.zeros(32, 28), is_train=True)
        self.target = torch.randn(32, 28)

    def forward(self, model):
        seed(2026)
        return model(dict(self.batch))

    def test_legacy_default_a0_a1_exact_forward_and_buffers(self):
        models = [copy.deepcopy(self.model) for _ in range(4)]
        enc = models[0].a2c_network.encoder
        enc.forward = types.MethodType(self.legacy, enc)
        models[2].a2c_network.encoder._latent_align_config["direction"] = "bidirectional"
        models[3].a2c_network.encoder._latent_align_config["direction"] = "post_to_prior"
        outputs = [self.forward(model) for model in models]
        for result in outputs[1:]:
            for key in ("mus", "post_mu", "prior_mu", "vae_kl_loss", "vae_commit_loss"):
                self.assertTrue(torch.equal(outputs[0][key], result[key]), key)
                self.assertTrue(torch.isfinite(result[key]).all(), key)
        for model in models[1:]:
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_real_loss_gradients(self):
        for direction in ("bidirectional", "post_to_prior"):
            for loss_kind in ("align", "action", "commit", "full"):
                model = copy.deepcopy(self.model)
                model.a2c_network.encoder._latent_align_config["direction"] = direction
                out = self.forward(model)
                action = (out["mus"] - self.target).pow(2).sum(-1).mean()
                losses = dict(align=out["vae_kl_loss"], action=action,
                              commit=out["vae_commit_loss"])
                losses["full"] = 10 * action + 0.1 * losses["align"] + losses["commit"]
                losses[loss_kind].backward()
                actual = gradients(model)
                for name, value in actual.items():
                    expected = (loss_kind in ("align", "full") if "prior" in name
                                else not (loss_kind == "align" and direction == "post_to_prior"))
                    if expected:
                        self.assertGreater(value, 0, (direction, loss_kind, name))
                    else:
                        self.assertEqual(value, 0, (direction, loss_kind, name))
                print("RVQ gradients", direction, loss_kind, actual)

    def test_metrics_reach_model_and_are_detached(self):
        out = self.forward(self.model)
        for layer in range(8):
            for metric in ("entropy", "perplexity", "active_fraction"):
                key = "latent_align_code_%s_layer_%d" % (metric, layer)
                self.assertIn(key, out)
                self.assertFalse(out[key].requires_grad)
                self.assertTrue(torch.isfinite(out[key]))
        self.assertTrue(torch.equal(out["latent_align_loss"], out["vae_kl_loss"].detach()))

    def test_entropy_excludes_dropout_and_separates_layers(self):
        enc = self.model.a2c_network.encoder
        post = torch.zeros(4, 64)
        indices = torch.tensor([[0, 0, -1], [0, 1, -1], [1, -1, -1], [1, -1, -1]])
        stats = enc._calc_latent_align_stats(post, post, indices, torch.zeros(4))
        self.assertAlmostEqual(stats["latent_align_code_perplexity_layer_0"].item(), 2)
        self.assertAlmostEqual(stats["latent_align_code_perplexity_layer_1"].item(), 2)
        self.assertEqual(stats["latent_align_code_perplexity_layer_2"].item(), 0)
        self.assertEqual(stats["latent_align_code_active_fraction_layer_1"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()

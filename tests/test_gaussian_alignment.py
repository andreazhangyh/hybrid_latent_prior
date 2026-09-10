"""Phase 2-4 numerical, real-RVQ gradient, and checkpoint gates; no simulator."""
import copy
import io
import json
import math
import os
import unittest
from pathlib import Path

import torch

from isaacgymenvs.learning.cvae_network_builder import diagonal_gaussian_kl
from tests.test_latent_alignment_rvq import ROOT, make_model, seed, groups


def config(experiment):
    return dict(loss_type="gaussian_kl" if experiment in ("A2", "A3") else "mse",
                direction="post_to_prior" if experiment in ("A1", "A3") else "bidirectional",
                learn_prior_std=experiment in ("A2", "A3"))


def parameter_groups(model):
    result = groups(model)
    enc = model.a2c_network.encoder
    if hasattr(enc, "_prior_logstd_net"):
        result["_prior_logstd_net"] = list(enc._prior_logstd_net.parameters())
    return result


class GaussianAlignmentTests(unittest.TestCase):
    report = {}

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        if os.getenv("ALIGNMENT_REPORT"):
            path = Path(os.environ["ALIGNMENT_REPORT"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cls.report, indent=2) + "\n")

    def setUp(self):
        seed(42)
        self.batch = dict(obs=torch.randn(32, 105), goal_obs=torch.randn(32, 225),
                          prev_actions=torch.zeros(32, 28), is_train=True)
        self.target = torch.randn(32, 28)

    def test_fixed_equal_variance_identity_fp64(self):
        q = torch.randn(4, 64, dtype=torch.float64)
        p = torch.randn_like(q)
        std = torch.full_like(q, 0.3)
        actual = diagonal_gaussian_kl(q, std, p, std)
        expected = (q - p).square().sum(-1) / (2 * 0.3 ** 2)
        error = (actual - expected).abs().max().item()
        self.assertLess(error, 1e-6)
        self.report["equal_variance_max_abs_error"] = error
        self.assertLess(diagonal_gaussian_kl(q, std, q, std).abs().max().item(), 1e-12)
        larger = diagonal_gaussian_kl(2 * q - p, std, p, std)
        self.assertTrue((larger > actual).all())

    def test_optimal_variance(self):
        q = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
        p = torch.zeros_like(q)
        q_std = torch.full_like(q, 0.3)
        optimum = (q_std.square() + (q - p).square()).sqrt().requires_grad_()
        loss = diagonal_gaussian_kl(q, q_std, p, optimum).sum()
        grad, = torch.autograd.grad(loss, optimum)
        self.assertLess(grad.abs().max().item(), 1e-12)
        for scale in (0.8, 1.2):
            self.assertGreater(diagonal_gaussian_kl(q, q_std, p, optimum * scale).sum().item(), loss.item())

    def test_initialization_and_action_rvq_invariance(self):
        reference, reference_output = None, None
        for experiment in ("A0", "A1", "A2", "A3"):
            seed(314)
            model = make_model(alignment=config(experiment))
            if reference is None:
                reference = copy.deepcopy(model.state_dict())
            for key, value in reference.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), (experiment, key))
            enc = model.a2c_network.encoder
            if experiment in ("A2", "A3"):
                self.assertEqual(enc._prior_logstd_net.weight.abs().sum().item(), 0)
                self.assertTrue(torch.allclose(enc._prior_logstd_net.bias,
                                               torch.full_like(enc._prior_logstd_net.bias, math.log(0.3))))
            else:
                self.assertFalse(hasattr(enc, "_prior_logstd_net"))
            seed(9)  # All eight real RVQ layers active; dropout remains enabled.
            output = model(dict(self.batch))
            if reference_output is None:
                reference_output = output
                reference_buffers = copy.deepcopy(model.state_dict())
            for key in ("mus", "post_mu", "prior_mu", "vae_commit_loss"):
                self.assertTrue(torch.equal(output[key], reference_output[key]), (experiment, key))
            for key, value in reference_buffers.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), (experiment, key))
            prior = enc.encode_prior(self.batch["obs"])
            self.assertEqual(prior["sigma"].abs().sum().item(), 0)

    def test_four_gradient_matrices(self):
        self.report["gradient_l2"] = {}
        for experiment in ("A0", "A1", "A2", "A3"):
            self.report["gradient_l2"][experiment] = {}
            for kind in ("alignment", "action", "commitment", "full"):
                seed(314)
                model = make_model(alignment=config(experiment))
                model.zero_grad(set_to_none=True)
                seed(9)
                result = model(dict(self.batch))
                coefficient = 0.018 if experiment in ("A2", "A3") else 0.1
                losses = dict(alignment=result["vae_kl_loss"],
                              action=(result["mus"] - self.target).square().sum(-1).mean(),
                              commitment=result["vae_commit_loss"])
                losses["full"] = 10 * losses["action"] + coefficient * losses["alignment"] + losses["commitment"]
                losses[kind].backward()
                actual = {}
                for name, parameters in parameter_groups(model).items():
                    grads = [p.grad for p in parameters if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    norm = math.sqrt(sum(g.square().sum().item() for g in grads))
                    actual[name] = norm
                    expected = (kind in ("alignment", "full") if "prior" in name else
                                not (kind == "alignment" and experiment in ("A1", "A3")))
                    if expected:
                        self.assertGreater(norm, 0, (experiment, kind, name))
                    else:
                        self.assertEqual(norm, 0, (experiment, kind, name))
                self.report["gradient_l2"][experiment][kind] = actual
                print("GRADIENT_MATRIX", experiment, kind, actual)

    def test_fp32_clamped_extremes_and_quantdirect(self):
        for style in ("quantcond", "quantdirect"):
            model = make_model(alignment=config("A3"))
            enc = model.a2c_network.encoder
            enc._style = style
            for value in (-1e4, 1e4):
                with torch.no_grad():
                    enc._prior_logstd_net.bias.fill_(value)
                _, std = enc._calc_prior_std(torch.ones(2, 32, dtype=torch.float16))
                q = torch.full((2, 64), 1e4, requires_grad=True)
                p = torch.zeros_like(q, requires_grad=True)
                loss = enc._calc_latent_align_loss(q, p, std).mean()
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                model.zero_grad(set_to_none=True)
                loss.backward()
                self.assertIsNone(q.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            result = model(dict(self.batch))
            self.assertTrue(torch.isfinite(result["vae_kl_loss"]))

    def test_checkpoint_migration_roundtrip_and_rejection(self):
        old = make_model().state_dict()
        # A genuine pre-alignment state has no alignment metadata.
        for metadata in old._metadata.values():
            metadata.pop("latent_align", None)
        for experiment in ("A0", "A1", "A2", "A3"):
            model = make_model(alignment=config(experiment))
            expected_new = {key for key in model.state_dict() if key not in old}
            self.assertEqual(expected_new, ({"a2c_network.encoder._prior_logstd_net." + x
                                             for x in ("weight", "bias")} if experiment in ("A2", "A3") else set()))
            model.load_state_dict(old, strict=True)
            damaged = copy.deepcopy(old)
            del damaged["a2c_network.mu.weight"]
            with self.assertRaises(RuntimeError):
                model.load_state_dict(damaged, strict=True)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            saved = torch.load(stream)
            replica = make_model(alignment=config(experiment))
            replica.load_state_dict(saved, strict=True)
            seed(9)
            a = model(dict(self.batch))
            seed(9)
            b = replica(dict(self.batch))
            for key in ("mus", "vae_kl_loss", "vae_commit_loss"):
                self.assertTrue(torch.equal(a[key], b[key]), key)
            if experiment in ("A2", "A3"):
                with self.assertRaises(RuntimeError):
                    make_model().load_state_dict(saved, strict=False)
                del saved["a2c_network.encoder._prior_logstd_net.bias"]
                with self.assertRaises(RuntimeError):
                    replica.load_state_dict(saved)

    def test_real_legacy_checkpoint_full_network(self):
        checkpoint = ROOT / "isaacgymenvs/runs/imitation_hybrid_b6offnv3/nn/imitation_hybrid_b6offnv3_7500.pth"
        if not checkpoint.exists():
            self.skipTest("Local historical checkpoint is unavailable")
        state = torch.load(str(checkpoint), map_location="cpu")["model"]
        for experiment in ("A0", "A1", "A2", "A3"):
            make_model(full=True, alignment=config(experiment)).load_state_dict(state, strict=True)

    def test_hydra_presets(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / "isaacgymenvs/cfg"), version_base="1.1"):
            for experiment in ("A0", "A1", "A2", "A3"):
                cfg = compose(config_name="config", overrides=["task=LafanImitation",
                              "train=imitation/HybridDistill" + experiment])
                actual = cfg.train.params.config
                for key, value in config(experiment).items():
                    self.assertEqual(actual.latent_align[key], value)
                self.assertAlmostEqual(actual.vae_kl_loss_coef, 0.018 if experiment in ("A2", "A3") else 0.1)
                self.assertEqual(actual.max_epochs, 200000)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable in this process")
    def test_cuda_autocast_kl_is_fp32(self):
        model = make_model(alignment=config("A3")).cuda()
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in self.batch.items()}
        with torch.cuda.amp.autocast():
            out = model(batch)
        self.assertEqual(out["vae_kl_loss"].dtype, torch.float32)
        out["vae_kl_loss"].backward()
        enc = model.a2c_network.encoder
        self.assertTrue(torch.isfinite(enc._prior_logstd_net.weight.grad).all())
        self.assertGreater(enc._prior_logstd_net.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in enc._post_net.parameters()))


if __name__ == "__main__":
    unittest.main()

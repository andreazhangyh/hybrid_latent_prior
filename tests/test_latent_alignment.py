import unittest

import torch
import torch.nn as nn

from isaacgymenvs.learning.cvae_network_builder import ContinuousEncoder


class IdentityQuantizer(nn.Module):
    def forward(self, z):
        indices = torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device)
        commit_loss = z.new_zeros(())
        return z, indices, commit_loss


def init_weight(weight):
    nn.init.uniform_(weight, -0.2, 0.2)


def make_encoder(direction=None):
    latent_align = None
    if direction is not None:
        latent_align = {
            "loss_type": "mse",
            "direction": direction,
            "learn_prior_std": False,
            "reduction": "sum",
        }
    encoder = ContinuousEncoder(
        state_dim=5,
        goal_dim=7,
        latent_dim=3,
        units=[11, 13],
        activation=nn.Tanh(),
        initializer=init_weight,
        style="quantcond",
        kwargs={
            "quant_type": "simple",
            "code_num": 8,
            "num_quants": 1,
            "latent_align": latent_align,
        },
    )
    encoder._post_quantizer = IdentityQuantizer()
    return encoder


def grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += param.grad.detach().abs().sum().item()
    return total


def posterior_parameters(encoder):
    return list(encoder._post_net.parameters()) + list(encoder._post_loc_net.parameters())


def prior_parameters(encoder):
    return list(encoder._prior_net.parameters()) + list(encoder._prior_loc_net.parameters())


def assert_exact_tensor_equal(test_case, lhs, rhs):
    if not torch.equal(lhs, rhs):
        max_diff = (lhs - rhs).abs().max().item()
        test_case.fail("tensors differ; max abs diff = %g" % max_diff)


class LatentAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.obs = torch.randn(4, 5)
        self.goal = torch.randn(4, 7)

    def test_default_config_matches_explicit_a0(self):
        torch.manual_seed(123)
        default_encoder = make_encoder()
        explicit_a0 = make_encoder("bidirectional")
        explicit_a0.load_state_dict(default_encoder.state_dict())
        explicit_a0._post_quantizer = IdentityQuantizer()

        default_out = default_encoder(self.obs, self.goal)
        explicit_out = explicit_a0(self.obs, self.goal)

        for key in ["post_z", "prior_z", "kl_loss", "post_mu", "prior_mu"]:
            assert_exact_tensor_equal(self, default_out[key], explicit_out[key])

    def test_a0_a1_forward_alignment_loss_is_equal(self):
        torch.manual_seed(456)
        a0 = make_encoder("bidirectional")
        a1 = make_encoder("post_to_prior")
        a1.load_state_dict(a0.state_dict())
        a1._post_quantizer = IdentityQuantizer()

        a0_out = a0(self.obs, self.goal)
        a1_out = a1(self.obs, self.goal)

        for key in ["post_z", "prior_z", "kl_loss", "post_mu", "prior_mu"]:
            assert_exact_tensor_equal(self, a0_out[key], a1_out[key])

    def test_alignment_backward_routes_gradients_for_a0_and_a1(self):
        torch.manual_seed(789)
        a0 = make_encoder("bidirectional")
        a1 = make_encoder("post_to_prior")
        a1.load_state_dict(a0.state_dict())
        a1._post_quantizer = IdentityQuantizer()

        a0_out = a0(self.obs, self.goal)
        a0_out["kl_loss"].mean().backward()
        a0_posterior_grad = grad_norm(posterior_parameters(a0))
        a0_prior_grad = grad_norm(prior_parameters(a0))

        a1_out = a1(self.obs, self.goal)
        a1_out["kl_loss"].mean().backward()
        a1_posterior_grad = grad_norm(posterior_parameters(a1))
        a1_prior_grad = grad_norm(prior_parameters(a1))

        self.assertGreater(a0_posterior_grad, 0.0)
        self.assertGreater(a0_prior_grad, 0.0)
        self.assertEqual(a1_posterior_grad, 0.0)
        self.assertGreater(a1_prior_grad, 0.0)

    def test_action_like_loss_updates_a1_posterior_but_not_prior(self):
        torch.manual_seed(101112)
        a1 = make_encoder("post_to_prior")

        loss = a1(self.obs, self.goal)["post_z"].pow(2).mean()
        loss.backward()

        self.assertGreater(grad_norm(posterior_parameters(a1)), 0.0)
        self.assertEqual(grad_norm(prior_parameters(a1)), 0.0)


if __name__ == "__main__":
    unittest.main()

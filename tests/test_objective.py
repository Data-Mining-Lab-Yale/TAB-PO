"""Analytic checks of the actual loss used by the trainer; no model downloads."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
    from tabpo.objective import tabpo_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install training requirements to run objective checks")
class ObjectiveTests(unittest.TestCase):
    def loss(self, probs, **kwargs):
        token = torch.tensor([probs], dtype=torch.float32).log().requires_grad_()
        z = torch.tensor([0.0], requires_grad=True)
        ref = torch.tensor([0.0], requires_grad=True)
        total, parts = tabpo_loss(z, z, ref, ref, token, torch.ones_like(token, dtype=torch.bool),
                                  torch.ones_like(token), **kwargs)
        return total, parts, token, ref

    def test_initial_reference_margin_and_selected_nll(self):
        total, parts, _, _ = self.loss([0.5, 0.95])
        self.assertAlmostEqual(total.item(), math.log(2) + 0.5 * -math.log(0.5), places=6)
        self.assertEqual(parts["gate"].tolist(), [[1.0, 0.0]])

    def test_no_gated_tokens_has_zero_barrier(self):
        total, parts, _, _ = self.loss([0.9, 0.95])
        self.assertEqual(parts["barrier"].item(), 0)
        self.assertAlmostEqual(total.item(), math.log(2), places=6)

    def test_gradients_only_on_selected_tokens_and_no_reference_gradient(self):
        total, _, token, reference = self.loss([0.5, 0.95])
        total.backward()
        self.assertEqual(token.grad.tolist(), [[-0.5, 0.0]])
        self.assertIsNone(reference.grad)

    def test_zero_barrier_weight_reduces_to_dpo(self):
        total, _, token, _ = self.loss([0.1, 0.2], barrier_weight=0)
        total.backward()
        self.assertAlmostEqual(total.item(), math.log(2), places=6)
        self.assertEqual(token.grad.tolist(), [[0, 0]])

    def test_padding_and_prompt_do_not_contribute(self):
        logs = torch.tensor([[-100.0, math.log(0.5)]])
        z = torch.zeros(1)
        total, parts = tabpo_loss(z, z, z, z, logs, torch.tensor([[False, True]]), torch.ones_like(logs))
        self.assertAlmostEqual(parts["barrier"].item(), 0.5 * math.log(2), places=6)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            self.loss([0.5], tau=1)


if __name__ == "__main__":
    unittest.main()

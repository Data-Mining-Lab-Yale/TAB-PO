"""The TAB-PO preference and confidence-gated barrier equations."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def tabpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
               chosen_token_logps, chosen_mask, chosen_weights, *, beta=0.5, tau=0.87,
               barrier_weight=0.5, example_weights=None, gated_weight_normalization=False):
    """Return a scalar loss and per-example terms. Reference and gate are detached.

    Plain TAB-PO uses unit token weights and averages NLL over gated tokens.
    The auxiliary weighted/length-normalized variant averages over gated weight
    mass. Log probabilities must already be causally shifted and prompt-masked.
    """
    if beta <= 0 or not 0 < tau < 1 or barrier_weight < 0:
        raise ValueError("Require beta > 0, 0 < tau < 1, barrier_weight >= 0")
    margin = (policy_chosen - policy_rejected).float() - (reference_chosen.detach() - reference_rejected.detach()).float()
    preference = -F.logsigmoid(beta * margin)
    gate = ((chosen_token_logps.detach() < math.log(tau)) & chosen_mask.bool()).to(chosen_token_logps.dtype)
    numerator = (-chosen_token_logps * chosen_weights * gate).sum(dim=1)
    denominator = ((chosen_weights * gate).sum(dim=1).clamp(min=1e-6)
                   if gated_weight_normalization else gate.sum(dim=1).clamp(min=1.0))
    barrier = barrier_weight * numerator / denominator
    exw = torch.ones_like(preference) if example_weights is None else example_weights.to(preference).view(-1)
    if not torch.isfinite(exw).all() or (exw <= 0).any():
        raise ValueError("Example weights must be finite and positive")
    total = ((preference + barrier) * exw).sum() / exw.sum()
    return total, {"margin": margin, "preference": preference, "barrier": barrier, "gate": gate}

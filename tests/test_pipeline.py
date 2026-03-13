"""Integration tests for the Pre-MoE pipeline.

These tests verify the pipeline logic without requiring NCCL/multi-GPU.
"""

import torch
import pytest

from premoe.config import PreMoEConfig
from premoe.probe import LinearProbe
from premoe.dispatch_planner import compute_dispatch_plan, pack_tokens


def test_probe_topk_prediction():
    """Probe produces valid top-k expert indices."""
    probe = LinearProbe(hidden_dim=128, num_experts=64)
    probe.eval()

    h_pre = torch.randn(32, 128)
    with torch.no_grad():
        logits = probe(h_pre)
        pred_ids = torch.topk(logits, k=6, dim=-1).indices

    assert pred_ids.shape == (32, 6)
    assert pred_ids.min() >= 0
    assert pred_ids.max() < 64


def test_full_dispatch_plan_pipeline():
    """Test probe -> dispatch plan -> pack pipeline end-to-end."""
    config = PreMoEConfig(hidden_dim=128, num_experts=64, top_k=6, ep_size=2)
    probe = LinearProbe(config.hidden_dim, config.num_experts)
    probe.eval()

    h_pre = torch.randn(100, config.hidden_dim)

    with torch.no_grad():
        logits = probe(h_pre)
        pred_ids = torch.topk(logits, k=config.top_k, dim=-1).indices

    peer_indices, needs_peer = compute_dispatch_plan(
        pred_ids, config.num_experts, config.ep_size, rank=0
    )

    packed = pack_tokens(h_pre, peer_indices)
    assert packed.shape[0] == len(peer_indices)
    assert packed.shape[1] == config.hidden_dim


def test_probe_warm_started_from_gate():
    """Probe warm-started from gate weight achieves high recall."""
    num_experts = 64
    hidden_dim = 128
    topk = 6

    # Create a "gate weight" and use it as both gate and probe
    gate_w = torch.randn(num_experts, hidden_dim)

    probe = LinearProbe(hidden_dim, num_experts)
    with torch.no_grad():
        probe.linear.weight.copy_(gate_w)
    probe.eval()

    # Generate hidden states and compute true routing via gate
    h = torch.randn(200, hidden_dim)
    true_logits = h @ gate_w.T
    true_ids = torch.topk(true_logits, k=topk, dim=-1).indices

    # Probe predictions should match exactly (same weight)
    with torch.no_grad():
        pred_logits = probe(h)
        pred_ids = torch.topk(pred_logits, k=topk, dim=-1).indices

    # Should be identical since weights are the same
    match = (pred_ids == true_ids).all(dim=-1).float().mean()
    assert match == 1.0, f"Expected perfect match with identical weights, got {match}"


def test_dispatch_plan_symmetry():
    """rank=0 and rank=1 should dispatch complementary sets."""
    topk_ids = torch.tensor([
        [0, 1, 2, 3, 4, 32],   # needs both GPUs
        [0, 1, 2, 3, 4, 5],    # only GPU0
        [32, 33, 34, 35, 36, 37],  # only GPU1
    ])

    idx0, needs0 = compute_dispatch_plan(topk_ids, 64, 2, rank=0)
    idx1, needs1 = compute_dispatch_plan(topk_ids, 64, 2, rank=1)

    # Token 0: needs peer from both perspectives (has experts on both GPUs)
    assert needs0[0] and needs1[0]
    # Token 1: needs peer only from rank=1 (all on GPU0)
    assert not needs0[1] and needs1[1]
    # Token 2: needs peer only from rank=0 (all on GPU1)
    assert needs0[2] and not needs1[2]

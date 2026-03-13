"""Unit tests for dispatch planning logic."""

import torch
import pytest

from premoe.dispatch_planner import compute_dispatch_plan, pack_tokens, verify_dispatch


class TestComputeDispatchPlan:
    """Tests for compute_dispatch_plan."""

    def test_all_local(self):
        """All experts on local GPU -> no tokens need peer dispatch."""
        # EP=2, 64 experts -> experts 0-31 on GPU0, 32-63 on GPU1
        # All top-k experts are on GPU0 (rank=0)
        topk_ids = torch.tensor([[0, 1, 2, 3, 4, 5],
                                  [10, 11, 12, 13, 14, 15]])
        peer_idx, needs_peer = compute_dispatch_plan(topk_ids, 64, 2, rank=0)
        assert len(peer_idx) == 0
        assert not needs_peer.any()

    def test_all_remote(self):
        """All experts on peer GPU -> all tokens need dispatch."""
        # All experts are on GPU1 (rank=0 perspective)
        topk_ids = torch.tensor([[32, 33, 34, 35, 36, 37],
                                  [40, 41, 42, 43, 44, 45]])
        peer_idx, needs_peer = compute_dispatch_plan(topk_ids, 64, 2, rank=0)
        assert len(peer_idx) == 2
        assert needs_peer.all()

    def test_mixed(self):
        """Some experts local, some remote -> only affected tokens dispatched."""
        topk_ids = torch.tensor([
            [0, 1, 2, 3, 4, 5],      # all local
            [0, 1, 2, 3, 4, 32],     # one remote -> needs peer
            [32, 33, 34, 35, 36, 37], # all remote
        ])
        peer_idx, needs_peer = compute_dispatch_plan(topk_ids, 64, 2, rank=0)
        assert len(peer_idx) == 2
        assert not needs_peer[0]
        assert needs_peer[1]
        assert needs_peer[2]

    def test_empty_input(self):
        """Empty input returns empty results."""
        topk_ids = torch.zeros(0, 6, dtype=torch.long)
        peer_idx, needs_peer = compute_dispatch_plan(topk_ids, 64, 2, rank=0)
        assert len(peer_idx) == 0
        assert len(needs_peer) == 0

    def test_rank1_perspective(self):
        """Same experts, but from rank=1 perspective (reversed)."""
        topk_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])  # all on GPU0
        peer_idx, needs_peer = compute_dispatch_plan(topk_ids, 64, 2, rank=1)
        # From rank=1, experts 0-31 are remote
        assert len(peer_idx) == 1
        assert needs_peer[0]


class TestPackTokens:
    """Tests for pack_tokens."""

    def test_pack_subset(self):
        """Pack specific token indices."""
        hidden = torch.randn(10, 128)
        indices = torch.tensor([2, 5, 7])
        packed = pack_tokens(hidden, indices)
        assert packed.shape == (3, 128)
        assert torch.allclose(packed[0], hidden[2])
        assert torch.allclose(packed[1], hidden[5])

    def test_pack_empty(self):
        """Empty indices returns empty tensor."""
        hidden = torch.randn(10, 128)
        indices = torch.tensor([], dtype=torch.long)
        packed = pack_tokens(hidden, indices)
        assert packed.shape == (0, 128)

    def test_pack_contiguous(self):
        """Packed output is contiguous."""
        hidden = torch.randn(10, 128)
        indices = torch.tensor([1, 3, 5])
        packed = pack_tokens(hidden, indices)
        assert packed.is_contiguous()


class TestVerifyDispatch:
    """Tests for verify_dispatch."""

    def test_perfect_match(self):
        """Identical predictions -> no mismatches."""
        pred = torch.tensor([[0, 1, 2, 3, 4, 5]])
        true = torch.tensor([[0, 1, 2, 3, 4, 5]])
        mismatches = verify_dispatch(pred, true, 64, 2, rank=0)
        assert len(mismatches) == 0

    def test_gpu_level_match(self):
        """Different experts but same GPU destinations -> no mismatch."""
        # Both have all experts on GPU0
        pred = torch.tensor([[0, 1, 2, 3, 4, 5]])
        true = torch.tensor([[6, 7, 8, 9, 10, 11]])
        mismatches = verify_dispatch(pred, true, 64, 2, rank=0)
        assert len(mismatches) == 0

    def test_gpu_level_mismatch(self):
        """Different GPU destinations -> mismatch detected."""
        pred = torch.tensor([[0, 1, 2, 3, 4, 5]])   # all GPU0
        true = torch.tensor([[0, 1, 2, 3, 4, 32]])  # one on GPU1
        mismatches = verify_dispatch(pred, true, 64, 2, rank=0)
        assert len(mismatches) == 1

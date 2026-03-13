"""Unit tests for LinearProbe and probe loading."""

import tempfile
from pathlib import Path

import torch
import pytest

from premoe.probe import LinearProbe, load_probes
from premoe.config import PreMoEConfig


def test_probe_forward_shape():
    """Probe output shape matches [N, num_experts]."""
    probe = LinearProbe(hidden_dim=2048, num_experts=64)
    x = torch.randn(128, 2048)
    out = probe(x)
    assert out.shape == (128, 64)


def test_probe_forward_batch_sizes():
    """Probe handles different batch sizes including 1 and 0."""
    probe = LinearProbe(hidden_dim=2048, num_experts=64)

    out1 = probe(torch.randn(1, 2048))
    assert out1.shape == (1, 64)

    out0 = probe(torch.randn(0, 2048))
    assert out0.shape == (0, 64)


def test_probe_weight_shape():
    """Probe weight has shape [num_experts, hidden_dim]."""
    probe = LinearProbe(hidden_dim=2048, num_experts=64)
    assert probe.linear.weight.shape == (64, 2048)


def test_load_probes():
    """Test loading probes from disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake probe weight file
        probe = LinearProbe(hidden_dim=128, num_experts=8)
        torch.save(probe.state_dict(), Path(tmpdir) / "probe_layer1.pt")

        config = PreMoEConfig(
            hidden_dim=128,
            num_experts=8,
            probe_dir=tmpdir,
            anchor_layers=[1],
        )

        loaded = load_probes(config, device=torch.device("cpu"), dtype=torch.float32)
        assert 1 in loaded
        assert loaded[1].linear.weight.shape == (8, 128)


def test_load_probes_missing():
    """Test that missing probe files are handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PreMoEConfig(
            hidden_dim=128,
            num_experts=8,
            probe_dir=tmpdir,
            anchor_layers=[1, 14, 26],
        )
        loaded = load_probes(config, device=torch.device("cpu"))
        assert len(loaded) == 0


def test_probe_deterministic():
    """Probe in eval mode produces deterministic output."""
    probe = LinearProbe(hidden_dim=128, num_experts=8)
    probe.eval()
    x = torch.randn(32, 128)
    out1 = probe(x)
    out2 = probe(x)
    assert torch.allclose(out1, out2)
